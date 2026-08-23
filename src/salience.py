"""Routed-salience profiling THROUGH the pager — no checkpoint required.

For a `full-memmap` artifact the original checkpoint may be gone (consumed at
split time); the memmaps ARE the model. This tool spies on the exact-mode
forward to accumulate per-(layer, expert, neuron) E[h^2] over a calibration
corpus, combines it with ||W_down[:, i]||^2 read from the planes, and writes
standard salience orders — usable as the pool prior and as input to floor.py.

Cost: ~2x the exact-mode forward (the spy recomputes gate/up), plus one
dequant pass per (layer, expert) for the weight norms.

Usage:
    python src/salience.py --artifact artifacts/qwen80-stream \
        --data data/calib.jsonl --tokens 6000 --out runs/orders.npz
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

import pager
from common import GROUP, find_moe_blocks, load_corpus, route, set_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Routed salience via the pager")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokens", type=int, default=6000)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from pathlib import Path

    set_seeds(1234)
    model, tokenizer = pager.load_model(args.artifact)
    pager.patch_pool(model, Path(args.artifact), pool_c=8, pool_k=2)
    pager.S["mode"] = "exact"

    blocks = find_moe_blocks(model)
    layer_of = {id(b): li for li, b in blocks}
    st0 = pager.S["layers"][id(blocks[0][1])]
    n_exp = st0.n_experts
    inter = st0.mm["gate_proj"]["out"]
    print(f"[salience] {len(blocks)} layers · {n_exp} experts · width {inter}")

    acc = {li: np.zeros((n_exp, inter), dtype=np.float64) for li, _ in blocks}
    cnt = {li: np.zeros(n_exp, dtype=np.int64) for li, _ in blocks}
    block_cls = type(blocks[0][1])
    orig_call = block_cls.__call__

    def spy(self, x):
        st = pager.S["layers"][id(self)]
        li = layer_of[id(self)]
        x_flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        inds, _ = route(self, x_flat)
        inds_np = np.array(inds)
        uniq, inv = np.unique(inds_np, return_inverse=True)
        rme = mx.array(inv.reshape(inds_np.shape).astype(np.int32))
        xx = mx.expand_dims(x_flat, (-2, -3))

        def pfx(proj):
            r0e = st._rows(proj, "p0", list(uniq))
            r1e = st._rows(proj, "p1", list(uniq))
            s_e, b_e = st._sb(proj, list(uniq))
            return (mx.gather_qmm(xx, r0e, s_e * 4, b_e, rhs_indices=rme,
                                  transpose=True, group_size=GROUP, bits=2)
                    + mx.gather_qmm(xx, r1e, s_e, mx.zeros_like(s_e),
                                    rhs_indices=rme, transpose=True,
                                    group_size=GROUP, bits=2))

        h = self.switch_mlp.activation(pfx("up_proj"), pfx("gate_proj"))
        h2 = (h.astype(mx.float32) ** 2).squeeze(-2)
        mx.eval(h2)
        h2n = np.array(h2).reshape(-1, h2.shape[-1])
        flat = inds_np.reshape(-1)
        np.add.at(acc[li], flat, h2n)
        np.add.at(cnt[li], flat, 1)
        return orig_call(self, x)

    block_cls.__call__ = spy
    rows = load_corpus(Path(args.data), args.tokens)
    served = 0
    t0 = time.perf_counter()
    try:
        for i, row in enumerate(rows):
            ids = mx.array(tokenizer.encode(row["text"])[: args.chunk_len])[None]
            mx.eval(model(ids))
            served += ids.shape[1]
            mx.clear_cache()
            if (i + 1) % 3 == 0:
                print(f"  {served} tokens ({time.perf_counter() - t0:.0f}s)")
    finally:
        block_cls.__call__ = orig_call

    result = {}
    for li, blk in blocks:
        st = pager.S["layers"][id(blk)]
        wnorm = np.zeros((n_exp, inter), dtype=np.float64)
        for e in range(n_exp):
            r0 = st._rows("down_proj", "p0", [e])[0]
            r1 = st._rows("down_proj", "p1", [e])[0]
            s_e, b_e = st._sb("down_proj", [e])
            wd = (mx.dequantize(r0, s_e[0] * 4, b_e[0], group_size=GROUP, bits=2)
                  + mx.dequantize(r1, s_e[0], mx.zeros_like(s_e[0]),
                                  group_size=GROUP, bits=2))
            mx.eval(wd)
            wnorm[e] = np.array((wd.astype(mx.float32) ** 2).sum(axis=0))
            del wd
        h2m = acc[li] / np.maximum(cnt[li], 1)[:, None]
        result[f"salience_{li}"] = (h2m * wnorm).astype(np.float32)
        mx.clear_cache()
        if li % 10 == 0:
            print(f"  layer {li} done")
    np.savez_compressed(args.out, **result)
    print(f"[out] {args.out}  ({served} tokens)")


if __name__ == "__main__":
    main()
