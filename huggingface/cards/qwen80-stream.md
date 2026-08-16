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
| 4-bit size / served peak | 42 GB / **16.5–17.4 GB** |
| Decode speed | 20–21.5 tok/s (indicative) |
| HumanEval / GSM8K (greedy, n=25) | 84% / **96%** — the best GSM8K on this machine |
| True base (exact-mode paging, 4.3 GB peak) | PPL 2.228 code / 5.557 general — the strongest base this hardware has touched |
| Serving toll (pool+drop, teacher-forced) | +28% code / **+120% general** |

## Honest limits — read before choosing this model

Under this runtime the 80B is a **reasoning specialist, not a generalist**:
the pool's drop policy is nearly free in task-style decode (GSM8K 96%) but
expensive in dispersed domains (the +120% general-PPL toll is measured, not
hypothetical — its knowledge advantage evaporates in broad-knowledge
serving). For a balanced daily driver at this memory, prefer the
[35B artifact](https://huggingface.co/jesusluque/qwen3.5-35b-topiary-stream).
n=25 task statistics: ≤2-item differences are indistinguishable.

## Provenance

Split deterministically from mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit
with `split.py --layout full-memmap`. Routed-salience orders (17.4k calibration
tokens) included as the pool prior. Full measurement history:
https://github.com/jesusluque/topiary-stream
