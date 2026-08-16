"""Build a universal 2D floor: P0 x salience-prefix, resident, for `floor2d`.

Every expert contributes its top-k_floor neurons (by routed salience) at the
P0 (2-bit) level, packed into one resident safetensors. With the floor loaded,
no slot is ever dropped: out-of-pool experts serve their floor slice.

Honest guidance from the measurements: floor quality tracks salience
concentration. On a model whose per-expert prefix captures >85-90% of energy
the floor is a real quality tier; at ~50% capture (flat, load-balanced
salience) it degrades to a coherence-of-last-resort and may not be servable —
measure your model's curve first (salience.py prints it implicitly; sort and
cumsum the orders).

Usage:
    python src/floor.py --artifact artifacts/qwen80-stream \
        --orders runs/orders.npz --k-floor 256 --out artifacts/qwen80-floor256.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import GROUP

PARTS = ("gate_proj", "up_proj", "down_proj")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the universal 2D floor")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--orders", required=True)
    parser.add_argument("--k-floor", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    kf = args.k_floor
    assert kf % GROUP == 0
    art = Path(args.artifact)

    orders = np.load(args.orders)
    manifest = json.load(open(art / "stream_manifest.json"))
    layers = sorted({m["layer"] for m in manifest.values()})
    out_tensors: dict[str, mx.array] = {}
    total = 0
    for li in layers:
        sal = orders[f"salience_{li}"]                    # [E, inter]
        pref = np.argsort(-sal, axis=1)[:, :kf]
        pref.sort(axis=1)   # natural order keeps down-column slicing coherent
        for proj in PARTS:
            m = manifest[f"L{li}.{proj}"]
            e, outd = m["experts"], m["out"]
            p0 = np.memmap(art / f"L{li}.{proj}.p0.bin", dtype=np.uint32,
                           mode="r", shape=(e, m["words"]))
            sb = np.memmap(art / f"L{li}.{proj}.sb.bin", dtype=np.float16,
                           mode="r", shape=(e, 2, m["sb_cols"]))
            words_row = m["words"] // outd
            groups_row = m["sb_cols"] // outd
            p0m = np.asarray(p0).reshape(e, outd, words_row)
            sbm = np.asarray(sb).reshape(e, 2, outd, groups_row)
            if proj != "down_proj":
                w = np.take_along_axis(p0m, pref[:, :, None], axis=1)
                s = np.take_along_axis(sbm[:, 0], pref[:, :, None], axis=1)
                b = np.take_along_axis(sbm[:, 1], pref[:, :, None], axis=1)
            else:
                blocks = pref[:, ::GROUP] // GROUP
                widx = (blocks[:, :, None] * (GROUP * 2 // 32)
                        + np.arange(GROUP * 2 // 32)).reshape(e, -1)
                w = np.take_along_axis(p0m, widx[:, None, :].repeat(outd, 1), axis=2)
                s = np.take_along_axis(sbm[:, 0], blocks[:, None, :].repeat(outd, 1), axis=2)
                b = np.take_along_axis(sbm[:, 1], blocks[:, None, :].repeat(outd, 1), axis=2)
            for name, arr, dt in (("w", w, mx.uint32), ("s", s, mx.float16),
                                  ("b", b, mx.float16)):
                t = mx.array(np.ascontiguousarray(arr)).astype(dt)
                out_tensors[f"L{li}.{proj}.{name}"] = t
                total += t.nbytes
        out_tensors[f"L{li}.pref"] = mx.array(pref.astype(np.int32))
        if li % 20 == 0:
            print(f"  layer {li} ({total / 1e9:.2f} GB)")
    mx.save_safetensors(args.out, out_tensors)
    print(f"[out] {args.out}  ({total / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
