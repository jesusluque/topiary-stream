"""Protección de tensores sensibles (lo que Unsloth hace por tensor, dentro
de nuestra pirámide): el ROUTER a precisión completa (BF16 oficial) y el
esqueleto no-experto (atención/embeddings/cabeza) a 8 bits, tomados por
RANGOS HTTP de los safetensors remotos — sin descargar el checkpoint entero.
Mecanismo: overrides por ruta en config["quantization"] (False = sin
cuantizar; {"bits":8,...} = 8 bits), que mlx-lm ya aplica por capa.

Usage:
    python src/protect.py --artifact artifacts/qwen80-stream --router \
        --router-repo Qwen/Qwen3-Next-80B-A3B-Instruct
    python src/protect.py --artifact artifacts/qwen80-stream --skeleton8 \
        --skel-repo mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url
from safetensors.numpy import load_file, save_file

DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32,
      "I32": np.int32, "U8": np.uint8}


class RemoteShard:
    """Cabecera + lectura por rango de un safetensors en HF."""

    def __init__(self, repo: str, fname: str):
        self.url = hf_hub_url(repo, fname)
        r = requests.get(self.url, headers={"Range": "bytes=0-7"}, allow_redirects=True, timeout=60)
        r.raise_for_status()
        n = struct.unpack("<Q", r.content[:8])[0]
        r = requests.get(self.url, headers={"Range": f"bytes=8-{8 + n - 1}"}, allow_redirects=True, timeout=120)
        self.header = json.loads(r.content)
        self.base = 8 + n

    def read(self, name: str) -> np.ndarray:
        meta = self.header[name]
        a, b = meta["data_offsets"]
        r = requests.get(self.url, headers={"Range": f"bytes={self.base + a}-{self.base + b - 1}"},
                         allow_redirects=True, timeout=600)
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=DT[meta["dtype"]]).reshape(meta["shape"])
        return arr, meta["dtype"]


def fetch(repo: str, names: list[str]) -> dict:
    idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
    shards: dict[str, RemoteShard] = {}
    out = {}
    for i, n in enumerate(names):
        fname = idx[n]
        if fname not in shards:
            shards[fname] = RemoteShard(repo, fname)
        out[n] = shards[fname].read(n)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(names)} tensores")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--router", action="store_true", help="router a BF16 desde el repo oficial")
    ap.add_argument("--router-repo", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    ap.add_argument("--skeleton8", action="store_true", help="no-expertos a 8 bits desde un repo 8bit")
    ap.add_argument("--skel-repo", default="mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit")
    args = ap.parse_args()

    art = Path(args.artifact)
    cfg = json.load(open(art / "config.json"))
    q = cfg.setdefault("quantization", {})
    sk = load_file(str(art / "model.safetensors"))
    n_layers = cfg.get("num_hidden_layers", 48)

    if args.router:
        names = [f"model.layers.{i}.mlp.gate.weight" for i in range(n_layers)]
        print(f"[router] {len(names)} routers por rango desde {args.router_repo}")
        got = fetch(args.router_repo, names)
        for i in range(n_layers):
            base = f"model.layers.{i}.mlp.gate"
            arr, dt = got[f"{base}.weight"]
            sk[f"{base}.weight"] = arr                  # BF16 crudo (uint16 view)
            sk.pop(f"{base}.scales", None); sk.pop(f"{base}.biases", None)
            q[base] = False                             # sin cuantizar al cargar
        print("[router] sustituidos (BF16, sin cuantizar)")

    if args.skeleton8:
        # todos los tensores cuantizados no-experto del esqueleto
        quant_paths = sorted({k.rsplit(".", 1)[0] for k in sk if k.endswith(".scales")
                              and "switch_mlp" not in k and ".mlp.gate" not in k})
        names = [f"{p}.{s}" for p in quant_paths for s in ("weight", "scales", "biases")]
        print(f"[skeleton8] {len(quant_paths)} módulos ({len(names)} tensores) desde {args.skel_repo}")
        got = fetch(args.skel_repo, names)
        for p in quant_paths:
            for s in ("weight", "scales", "biases"):
                sk[f"{p}.{s}"] = got[f"{p}.{s}"][0]
            q[p] = {"group_size": 64, "bits": 8}
        print("[skeleton8] sustituidos (8 bits)")

    # guardar: safetensors con dtypes preservados (BF16 como uint16 → marcar)
    meta = {}
    tensors = {}
    for k, v in sk.items():
        tensors[k] = v
    save_file(tensors, str(art / "model.safetensors"), metadata={"format": "mlx"})
    json.dump(cfg, open(art / "config.json", "w"), indent=2)
    print("[ok] esqueleto y config reescritos")


if __name__ == "__main__":
    main()
