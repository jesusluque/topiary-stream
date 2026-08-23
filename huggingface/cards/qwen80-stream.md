---
license: apache-2.0
base_model: Qwen/Qwen3-Next-80B-A3B-Instruct
library_name: mlx
tags:
- mlx
- moe
- topiary-stream
- quantized
- bit-planes
---

# qwen3-next-80b-topiary-stream — Topiary Stream artifact (full-memmap)

An 80B MoE (42 GB at 4-bit — 1.8× this machine's RAM) served on 24 GB Apple
Silicon: a 1.37 GB expert-free skeleton plus row-pageable bit-plane memmaps
for all 512 experts × 48 layers, driven by a unified per-layer pool (C experts
at the P0 floor, K of them completed to exact 4-bit), seeded by a
routed-salience traffic prior (included).

**Serving caveat:** requires the
[Topiary Stream runtime](https://github.com/jesusluque/topiary-stream) — this
does not load with stock `mlx_lm`:

```bash
python src/serve.py --artifact jesusluque/qwen3-next-80b-topiary-stream \
    --pool-c 240 --pool-k 32 --orders orders_routed.npz
```

## Measured results (Apple M5 Pro, 24 GB)

| Metric | Value |
|---|---|
| 4-bit size / served peak | 42 GB / **16.5–17.7 GB** |
| Decode speed | 20–21.5 tok/s (indicative) |
| HumanEval / GSM8K (greedy, n=15, exact-prefill serving) | **15/15 / 15/15** |
| Served PPL (exact prefill) | 2.2280 code / 5.5569 general — **equals the true base** to four decimals |
| True base (exact-mode paging, 4.3 GB peak) | PPL 2.228 code / 5.557 general — the strongest base this hardware has touched |
| MATH-500 / MBPP (greedy, n=100, C=240 K=32, refresh 128) | 65% / 81% |
| MMLU / LAMBADA (n=500) | **85.2%** / **78.2%** |
| Decode-regime KLD vs exact base (WikiText 4×512, live cache) | 0.774 mean / p99 9.0 at refresh 256; 0.303 at refresh 32 (−47% tok/s) |
| Same 80B as Unsloth UD-Q2_K_XL (30.1 GB, llama.cpp CPU-only, 12.6 tok/s) | rival 69 / 86 / 86.2 on MATH / MBPP / MMLU; rival KLD 0.195 |

## Honest limits — read before choosing this model

The runtime serves **prefill exact by design** (the prompt is one batched
pass; the needed P1 planes read once from the memmaps), so the pool policy
governs decode only. Before that design fix, pool-served prefill cost +28%
code / +120% general PPL and tasks measured 84%/96% — the prefill toll, not
the decode policy, was the handicap. With exact prefill the served model
matches its true base on PPL and scores 15/15 on both task sets (n=15;
small-n statistics — treat as "no detected degradation", not proof of
equality). Decode remains gate-governed: a pool miss serves the P0 floor for
one token.

**The decode-side toll is real and measured.** Against a calibrated static
2-bit of the same model (Unsloth UD-Q2_K_XL), this artifact is at par on
knowledge (MMLU 85.2 vs 86.2, n.s.), ~5 points behind on generative tasks
(n=100, directional), and clearly behind on general-prose fidelity (KLD
0.774 vs 0.195 — a pool miss drops an expert; a calibrated 2-bit never
does). What it buys is system-level: it fits in 17 GB where the static needs
30, runs 40–70% faster on the GPU, serves the prompt exact (teacher-forced KLD 0.000), and has a
floor and an elastic governor. Choose it for tasks and interactive use on a
24 GB machine; do not choose it for long general-prose fidelity.

## Provenance

Split deterministically from mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit
with `split.py --layout full-memmap`. Routed-salience orders (17.4k calibration
tokens) included as the pool prior. Full measurement history:
https://github.com/jesusluque/topiary-stream
