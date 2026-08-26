"""Split a quantized MoE checkpoint into a servable Topiary-Stream artifact.

Two layouts, one command:

  --layout resident-p0   (flagship; model's P0 must fit in RAM)
      The checkpoint keeps P0 (high 2 bits of every 4-bit expert code) in the
      weight slots (config gains per-module bits=2 overrides) and the low
      planes go to per-(layer, proj) `*.p1.bin` memmaps. Serve with
      `serve.py` (fast-path + governor). RAM at load ≈ 60% of the 4-bit model.

  --layout full-memmap   (models whose P0 alone does not fit)
      Nothing expert-shaped stays in the checkpoint: P0, P1 AND scales/biases
      go to row-pageable memmaps, and a tiny skeleton checkpoint (non-expert
      weights + 1x64 dummy switch tensors + `stream_skeleton` config flag) is
      written alongside. Serve with `serve.py --pool-c ...` (unified pool).

The artifact directory is self-contained (config/tokenizer included): it is
what you upload, download and serve.

`--consume` deletes each source shard (symlink AND blob for HF caches) after
processing, capping peak disk at roughly checkpoint + one shard of planes.
Only use it when the artifact replaces the checkpoint; the skeleton (or the
resident-p0 checkpoint) is written FIRST so the runtime never needs the
consumed source again — a lesson paid for with a 132 GB re-download.

Usage:
    python src/split.py --src mlx-community/Qwen3.5-35B-A3B-4bit \
        --out artifacts/qwen35-stream --layout resident-p0
    python src/split.py --src mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit \
        --out artifacts/qwen80-stream --layout full-memmap --consume
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import GROUP, pack, pack2, unpack4

PARTS = ("gate_proj", "up_proj", "down_proj")
AUX = ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
       "chat_template.jinja", "special_tokens_map.json", "vocab.json", "merges.txt")
DUMMY_INTER = 64


def resolve_src(src: str) -> Path:
    if Path(src).exists():
        return Path(src)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(src))


def is_switch_weightlike(key: str) -> bool:
    parts = key.split(".")
    return ".switch_mlp." in key and len(parts) >= 2 and parts[-2] in PARTS


def layer_of(key: str) -> int:
    return int(key.split(".layers.")[1].split(".")[0])


def consume_shard(shard_path: str) -> None:
    p = Path(shard_path)
    target = p.resolve()
    p.unlink()
    if target != p and target.exists():
        target.unlink()


# ------------------------------------------------------------- resident-p0


def split_resident_p0(src: Path, out: Path, consume: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    p0_prefixes: set[str] = set()
    index = json.load(open(src / "model.safetensors.index.json"))
    new_map: dict[str, str] = {}
    total_res = total_p1 = 0
    for shard_path in sorted(glob.glob(str(src / "model-*.safetensors"))):
        shard = Path(shard_path).name
        tensors = dict(mx.load(shard_path))
        out_tensors = {}
        for key in list(tensors.keys()):
            arr = tensors.pop(key)
            if is_switch_weightlike(key) and key.endswith(".weight"):
                li, proj = layer_of(key), key.split(".")[-2]
                codes = unpack4(arr)                     # [E, out, cols]
                p0 = pack2(codes >> 2)
                mx.eval(p0)
                out_tensors[key] = p0                    # P0 lives in the ckpt
                p0_prefixes.add(key[: -len(".weight")])
                e = codes.shape[0]
                p1 = pack2(codes & 3)
                np.asarray(p1).reshape(e, -1).tofile(out / f"L{li}.{proj}.p1.bin")
                manifest[f"L{li}.{proj}.p1.bin"] = {
                    "experts": e, "out": int(codes.shape[1]),
                    "cols": int(codes.shape[2]),
                    "words": int(np.asarray(p1).reshape(e, -1).shape[1]),
                    "layer": li, "proj": proj}
                total_p1 += p1.nbytes
                del codes, p1
            else:
                out_tensors[key] = arr
            new_map[key] = shard
        for v in out_tensors.values():
            total_res += v.nbytes
        mx.save_safetensors(str(out / shard), out_tensors)
        del tensors, out_tensors
        mx.clear_cache()
        if consume:
            consume_shard(shard_path)
        print(f"  {shard} done{' (consumed)' if consume else ''}")
    index["weight_map"] = new_map
    index["metadata"]["total_size"] = total_res
    json.dump(index, open(out / "model.safetensors.index.json", "w"))
    json.dump(manifest, open(out / "p1_manifest.json", "w"))
    cfg = json.load(open(src / "config.json"))
    for qkey in ("quantization", "quantization_config"):
        if qkey in cfg:
            for pref in sorted(p0_prefixes):
                cfg[qkey][pref] = {"group_size": GROUP, "bits": 2}
    cfg["stream_layout"] = "resident-p0"
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    for aux in AUX:
        if (src / aux).exists():
            shutil.copy(src / aux, out / aux)
    print(f"[out] {out}: resident {total_res / 1e9:.2f} GB · P1 memmaps {total_p1 / 1e9:.2f} GB")


# ------------------------------------------------------------- full-memmap


def dummy_switch(hidden: int, bits: int) -> dict[str, dict[str, mx.array]]:
    outd = {}
    for proj in PARTS:
        shape = ((1, DUMMY_INTER, hidden) if proj != "down_proj"
                 else (1, hidden, DUMMY_INTER))
        w, s, b = mx.quantize(mx.zeros(shape, dtype=mx.float32),
                              group_size=GROUP, bits=bits)
        mx.eval(w, s, b)
        outd[proj] = {"weight": w, "scales": s.astype(mx.bfloat16),
                      "biases": b.astype(mx.bfloat16)}
    return outd


def split_full_memmap(src: Path, out: Path, consume: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(src / "config.json"))
    tcfg = cfg.get("text_config", cfg)
    hidden = tcfg["hidden_size"]
    bits = int(cfg["quantization"]["bits"])
    dummies = dummy_switch(hidden, bits)

    manifest: dict[str, dict] = {}
    skeleton: dict[str, mx.array] = {}
    seen_layers: set[int] = set()
    total = 0
    shards = sorted(glob.glob(str(src / "model-*.safetensors")))
    # scales/biases de un mismo experto pueden vivir en shards distintos (checkpoints
    # antiguos): se recogen primero (son pequeños) y se consumen al ver las scales.
    sb_all: dict[str, mx.array] = {}
    for shard_path in shards:
        for key, arr in mx.load(shard_path).items():
            if is_switch_weightlike(key) and key.endswith((".scales", ".biases")):
                sb_all[key] = arr
    for shard_path in shards:
        tensors = mx.load(shard_path)
        done = 0
        for key in tensors:
            if is_switch_weightlike(key):
                li, proj = layer_of(key), key.split(".")[-2]
                tag = f"L{li}.{proj}"
                if key.endswith(".weight"):
                    codes = unpack4(tensors[key])
                    e, o, c = codes.shape
                    p0 = np.asarray(pack2(codes >> 2)).reshape(e, -1)
                    p1 = np.asarray(pack2(codes & 3)).reshape(e, -1)
                    p0.tofile(out / f"{tag}.p0.bin")
                    p1.tofile(out / f"{tag}.p1.bin")
                    manifest.setdefault(tag, {}).update(
                        {"experts": e, "out": o, "cols": c,
                         "words": int(p0.shape[1]), "layer": li, "proj": proj})
                    total += p0.nbytes + p1.nbytes
                    del codes, p0, p1
                    done += 1
                elif key.endswith(".scales"):
                    s = np.array(tensors[key].astype(mx.float16))
                    b = np.array(sb_all[key[: -len("scales")] + "biases"]
                                 .astype(mx.float16))
                    sb = np.stack([s.reshape(s.shape[0], -1),
                                   b.reshape(b.shape[0], -1)], axis=1)
                    sb.tofile(out / f"{tag}.sb.bin")
                    manifest.setdefault(tag, {})["sb_cols"] = int(sb.shape[2])
                    manifest[tag]["s_shape"] = list(s.shape[1:])
                    total += sb.nbytes
                if li not in seen_layers:
                    seen_layers.add(li)
                    prefix = key.split(".switch_mlp.")[0] + ".switch_mlp."
                    for proj_d, tt in dummies.items():
                        for leaf, arr in tt.items():
                            skeleton[f"{prefix}{proj_d}.{leaf}"] = arr
            else:
                skeleton[key] = tensors[key]
        mx.eval(*[v for v in skeleton.values()])
        del tensors
        mx.clear_cache()
        if consume:
            consume_shard(shard_path)
        print(f"  {Path(shard_path).name}: {done} expert tensors"
              f"{' (consumed)' if consume else ''}")

    sk_total = sum(v.nbytes for v in skeleton.values())
    mx.save_safetensors(str(out / "model.safetensors"), skeleton)
    index = {"metadata": {"total_size": sk_total},
             "weight_map": {k: "model.safetensors" for k in skeleton}}
    json.dump(index, open(out / "model.safetensors.index.json", "w"))
    cfg["stream_skeleton"] = True
    cfg["stream_layout"] = "full-memmap"
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    json.dump(manifest, open(out / "stream_manifest.json", "w"))
    for aux in AUX:
        if (src / aux).exists():
            shutil.copy(src / aux, out / aux)
    print(f"[out] {out}: skeleton {sk_total / 1e9:.2f} GB · memmaps {total / 1e9:.2f} GB "
          f"· {len(seen_layers)} MoE layers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a MoE checkpoint into a Stream artifact")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layout", required=True, choices=["resident-p0", "full-memmap"])
    parser.add_argument("--consume", action="store_true",
                        help="delete source shards (symlink+blob) after processing")
    args = parser.parse_args()
    src = resolve_src(args.src)
    if args.layout == "resident-p0":
        split_resident_p0(src, Path(args.out), args.consume)
    else:
        split_full_memmap(src, Path(args.out), args.consume)


if __name__ == "__main__":
    main()
