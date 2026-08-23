"""Serve a Topiary-Stream artifact: generation CLI with governor and stats.

Reads `stream_layout` from the artifact's config and dispatches:
  resident-p0  -> fastpath (P0 resident, P1 pool, optional elastic governor)
  full-memmap  -> pager (skeleton + unified pool; --serve-mode picks policy)

Double-prefill warm-up: the prompt runs once for routing statistics only,
pools adapt, then the real prefill runs against an informed pool. Without it
a cold uniform pool serves the prompt mostly degraded and the decode inherits
a poisoned KV (measured on a 235B).

Usage:
    python src/serve.py --artifact artifacts/qwen35-stream --pool-k 32 --governor
    python src/serve.py --artifact artifacts/qwen80-stream --pool-c 240 --pool-k 32
    python src/serve.py --artifact artifacts/qwen80-stream --serve-mode floor \
        --prompt "..."   # slow-but-never-degraded demonstration mode
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from common import set_seeds

DEFAULT_PROMPT = ("Write a Python function that merges two sorted lists into "
                  "one sorted list. Then explain its complexity and edge cases.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a Stream artifact")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--pool-k", type=int, default=32)
    parser.add_argument("--pool-c", type=int, default=240,
                        help="full-memmap only: experts per layer held at P0")
    parser.add_argument("--serve-mode", default="nosync",
                        choices=["nosync", "floor", "floor2d"],
                        help="full-memmap only (resident-p0 has its own policy)")
    parser.add_argument("--floor", default=None,
                        help="floor2d: universal floor safetensors (floor.py)")
    parser.add_argument("--orders", default=None,
                        help="full-memmap: routed-salience npz for the pool prior")
    parser.add_argument("--governor", action="store_true",
                        help="resident-p0: elastic memory governor")
    parser.add_argument("--gov-low", type=float, default=4.0,
                        help="governor: shrink K below this available-GB")
    parser.add_argument("--gov-high", type=float, default=7.0,
                        help="governor: grow K above this available-GB")
    parser.add_argument("--ema-mass", action="store_true",
                        help="EMA del pool ponderada por masa de gate (no por cuenta)")
    parser.add_argument("--prewarm", type=int, default=0,
                        help="precalentar el tier de desbordamiento con los N casi-elegidos (ranks k+1..)")
    parser.add_argument("--gear-sensor", default="miss", choices=["miss", "margin"],
                        help="sensor del gobernador de marchas: tasa de misses o margen top-k/top-k+1")
    parser.add_argument("--refresh-min-miss", type=float, default=0.0,
                        help="refresh selectivo: saltar capas con tasa de misses menor (0=todas)")
    parser.add_argument("--gear", default=None,
                        help="gobernador de dos marchas 'Clo:Klo,Chi:Khi' (p.ej. 240:32,290:1)")
    parser.add_argument("--gear-hi", type=float, default=0.25, help="tasa de misses que sube de marcha")
    parser.add_argument("--gear-lo", type=float, default=0.10, help="tasa de misses que baja de marcha")
    parser.add_argument("--refresh", type=int, default=256)
    parser.add_argument("--burst-len", type=int, default=0,
                        help="ráfaga de refresh en los primeros N tokens (0=no)")
    parser.add_argument("--burst-every", type=int, default=8)
    parser.add_argument("--ovf-merge", type=int, default=0,
                        help="refresh barato: N refreshes rápidos (tier de desbordamiento) por cada completo")
    parser.add_argument("--centroid", default="uniform",
                        choices=["uniform", "empirical"])
    parser.add_argument("--p1-frac", default="1.0")
    args = parser.parse_args()

    set_seeds(1234)
    layout = json.load(open(Path(args.artifact) / "config.json")).get("stream_layout")
    if layout == "resident-p0":
        import fastpath as rt
        from mlx_lm import load

        try:  # checkpoints con anchos per-layer (taper de topiary)
            from dense_loader import maybe_patch_per_layer
            maybe_patch_per_layer(args.artifact)
        except ImportError:
            pass
        model, tokenizer = load(args.artifact)
        mx.eval(model.parameters())
        print(f"[load] {mx.get_active_memory() / 1e9:.2f} GB resident")
        rt.patch_fast(model, Path(args.artifact), args.pool_k,
                      args.centroid, args.p1_frac)
        print(f"[pool] K={args.pool_k}/layer "
              f"({mx.get_active_memory() / 1e9:.2f} GB with pools)")
        governor = ((lambda: rt.govern(low=args.gov_low, high=args.gov_high))
                    if args.governor else None)
    elif layout == "full-memmap":
        import pager as rt

        model, tokenizer = rt.load_model(args.artifact)
        print(f"[load] {mx.get_active_memory() / 1e9:.2f} GB (skeleton)")
        rt.patch_pool(model, Path(args.artifact), args.pool_c, args.pool_k,
                      orders=args.orders)
        rt.S["mode"] = args.serve_mode
        rt.S["ovf_merge"] = args.ovf_merge
        rt.S["ema_mass"] = args.ema_mass
        rt.S["prewarm"] = args.prewarm
        rt.S["refresh_min_miss"] = args.refresh_min_miss
        rt.S["gear_sensor"] = args.gear_sensor
        if args.gear:
            (clo, klo), (chi, khi) = [tuple(int(x) for x in g.split(":")) for g in args.gear.split(",")]
            rt.S.update({"gear_cfg": {"lo": (clo, klo), "hi": (chi, khi)}, "gear": "lo",
                         "gear_hi_thr": args.gear_hi, "gear_lo_thr": args.gear_lo, "gear_dwell": 0})
        if args.serve_mode == "floor2d":
            from common import find_moe_blocks

            assert args.floor, "--serve-mode floor2d requires --floor"
            rt.load_floor(args.floor, find_moe_blocks(model))
        print(f"[pool] C={args.pool_c} K={args.pool_k} mode={args.serve_mode} "
              f"({mx.get_active_memory() / 1e9:.2f} GB)")
        governor = None
    else:
        raise SystemExit(f"not a Stream artifact (stream_layout={layout!r})")

    from mlx_lm.generate import stream_generate

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True)

    # double-prefill warm-up
    mx.eval(model(mx.array([list(prompt)])))
    rt.refresh_all()
    print("[warmup] pool adapted to the prompt")

    text, n = [], 0
    t_dec = None
    for r in stream_generate(model, tokenizer, prompt=prompt,
                             max_tokens=args.max_tokens):
        if t_dec is None:
            rt.refresh_all()
            t_dec = time.perf_counter()
        text.append(r.text)
        n += 1
        if (args.burst_len and n <= args.burst_len and n % args.burst_every == 0) or n % args.refresh == 0:
            if governor:
                msg = governor()
                if msg:
                    print(f"\n{msg} · {mx.get_active_memory() / 1e9:.1f} GB active",
                          flush=True)
                else:
                    from fastpath import available_gb

                    k_now = next(iter(rt.STATE["layers"].values())).pool_k
                    print(f"\n[gov] avail {available_gb():.1f} GB · K={k_now} (hold)",
                          flush=True)
            rt.refresh_all()
    dt = time.perf_counter() - t_dec
    print("".join(text))
    print(f"\n[serve] {n - 1} tokens in {dt:.1f}s = {(n - 1) / dt:.1f} tok/s "
          f"· peak {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
