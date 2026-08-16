"""Anchored precision pyramid: one artifact, three service levels.

Construction rule (the part that took an inversion to learn): the pyramid is
ANCHORED at the serving level, refined upward, truncated downward only under
gate protection. Deriving levels by truncating a fine master wins on uniform
weight metrics (L2 AND max error) yet loses end-to-end — the truncated grid
concentrates its error on salient weights. Measured on OLMoE: derived Q2 was
20x worse in PPL than a natively-fitted Q2; the anchored pyramid instead gives

  Q4  = native min/max fit           (exact serving quality)
  Q8  = anchor + refinement plane    (PPL 3.9209 vs true master 3.9213: free)
  Q2  = anchor truncated             (floor for gate-protected cold slots ONLY)

Each level is a standard affine tensor readable by stock MLX kernels.

Stages:
  anchored   8-bit master -> three checkpoints <out>-q{2,4,8}
  derive     naive top-down truncation (kept for reproducing the negative)
  requant    dequant -> native re-fit at N bits (baseline builder)

Usage:
    python src/pyramid.py --stage anchored --src <8bit-repo-or-dir> --out models/m
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import GROUP, pack, unpack_bits


def master_path(src: str) -> Path:
    if Path(src).exists():
        return Path(src)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(src))


def write_ckpt(src: Path, out: Path, transform, bits_out: int) -> None:
    """Apply `transform(prefix, wq, s, b) -> (wq', s', b')` to every quantized
    triple. Triples may be split across shards (old per-expert checkpoints):
    all shards are lazy-merged first and each transformed triple is written to
    its weight's shard."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(src / "config.json"))
    for qkey in ("quantization", "quantization_config"):
        if qkey in cfg:
            assert int(cfg[qkey].get("bits")) == 8, "master must be uniform 8-bit"
            overrides = {k: v for k, v in cfg[qkey].items() if isinstance(v, dict)}
            assert not overrides, f"per-module overrides unsupported: {list(overrides)[:3]}"
            cfg[qkey]["bits"] = bits_out

    index_p = src / "model.safetensors.index.json"
    index = json.load(open(index_p)) if index_p.exists() else None
    shards = sorted(glob.glob(str(src / "model*.safetensors")))
    allt: dict[str, mx.array] = {}
    home: dict[str, list[str]] = {}
    for shard_path in shards:
        shard = Path(shard_path).name
        d = mx.load(shard_path)
        allt.update(d)
        home[shard] = list(d.keys())
    quant = {k[: -len(".weight")] for k in allt
             if k.endswith(".weight") and k[: -len(".weight")] + ".scales" in allt}

    new_map: dict[str, str] = {}
    total = 0
    for shard_path in shards:
        shard = Path(shard_path).name
        out_tensors = {}
        for key in home[shard]:
            pref, _, leaf = key.rpartition(".")
            if pref in quant and leaf == "weight":
                wq, s2, b2 = transform(pref, allt[key], allt[pref + ".scales"],
                                       allt.get(pref + ".biases"))
                mx.eval(wq, s2, b2)
                out_tensors[key], out_tensors[pref + ".scales"] = wq, s2
                out_tensors[pref + ".biases"] = b2
            elif pref in quant and leaf in ("scales", "biases"):
                continue   # travel with their weight, whatever their shard
            else:
                out_tensors[key] = allt[key]
        for k, v in out_tensors.items():
            new_map[k] = shard
            total += v.nbytes
        mx.save_safetensors(str(out / shard), out_tensors)
        del out_tensors
        mx.clear_cache()
    if index is not None:
        index["weight_map"] = new_map
        index["metadata"]["total_size"] = total
        json.dump(index, open(out / "model.safetensors.index.json", "w"))
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    for aux in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                "special_tokens_map.json", "vocab.json", "merges.txt",
                "chat_template.jinja"):
        if (src / aux).exists():
            shutil.copy(src / aux, out / aux)
    print(f"[out] {out}  ({total / 1e9:.2f} GB)")


def stage_anchored(src: str, out_base: str) -> None:
    srcp = master_path(src)

    def tf4(pref, wq, s, b):
        wd = mx.dequantize(wq, s, b, group_size=GROUP, bits=8)
        return mx.quantize(wd, group_size=GROUP, bits=4)

    def tf8(pref, wq, s, b):
        wd = mx.dequantize(wq, s, b, group_size=GROUP, bits=8)
        q4, s4, b4 = mx.quantize(wd, group_size=GROUP, bits=4)
        mx.eval(q4, s4, b4)
        c4 = unpack_bits(q4, 4).astype(np.float32)
        w = np.array(wd, dtype=np.float32)
        s4n = np.repeat(np.array(s4, dtype=np.float32), GROUP, axis=-1)
        b4n = np.repeat(np.array(b4, dtype=np.float32), GROUP, axis=-1)
        r = w - (s4n * c4 + b4n)
        q_lo = np.clip(np.round((r + s4n / 2) / (s4n / 16) - 0.5), 0, 15)
        codes8 = (c4 * 16 + q_lo).astype(np.uint8)   # >>4 recovers Q4 exactly
        return pack(codes8, 8), s4 / 16, b4 - 15 * s4 / 32

    def tf2(pref, wq, s, b):
        wd = mx.dequantize(wq, s, b, group_size=GROUP, bits=8)
        q4, s4, b4 = mx.quantize(wd, group_size=GROUP, bits=4)
        mx.eval(q4, s4, b4)
        codes2 = (unpack_bits(q4, 4) >> 2).astype(np.uint8)
        return pack(codes2, 2), s4 * 4, b4 + 1.5 * s4

    for tag, tf, bits in (("q4", tf4, 4), ("q8", tf8, 8), ("q2", tf2, 2)):
        print(f"[anchored] level {tag}…")
        write_ckpt(srcp, Path(f"{out_base}-{tag}"), tf, bits)


def stage_derive(src: str, bits: int, out_dir: str) -> None:
    shift = 8 - bits

    def tf(pref, wq, s, b):
        codes = unpack_bits(wq, 8).astype(np.uint8) >> shift
        b2 = (b if b is not None else mx.zeros_like(s)) + s * (2 ** shift - 1) / 2
        return pack(codes, bits), s * (2 ** shift), b2

    write_ckpt(master_path(src), Path(out_dir), tf, bits)


def stage_requant(src: str, bits: int, out_dir: str) -> None:
    def tf(pref, wq, s, b):
        wd = mx.dequantize(wq, s, b, group_size=GROUP, bits=8)
        return mx.quantize(wd, group_size=GROUP, bits=bits)

    write_ckpt(master_path(src), Path(out_dir), tf, bits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchored precision pyramid")
    parser.add_argument("--stage", required=True,
                        choices=["anchored", "derive", "requant"])
    parser.add_argument("--src", required=True, help="uniform 8-bit master")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.stage == "anchored":
        stage_anchored(args.src, args.out)
    elif args.stage == "derive":
        stage_derive(args.src, args.bits, args.out)
    else:
        stage_requant(args.src, args.bits, args.out)


if __name__ == "__main__":
    main()
