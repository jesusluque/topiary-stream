---
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B
library_name: mlx
tags:
- mlx
- moe
- topiary-stream
- quantized
- bit-planes
---

# qwen3.5-35b-topiary-stream — Topiary Stream artifact (resident-p0)

A servable residency artifact for a model that does not fit whole in 24 GB:
the 4-bit checkpoint loads at 19.5 GB and Metal fails on generation. Every
expert is split into standard-kernel-readable bit planes (P0 floor resident
in the checkpoint, P1 completion in row-pageable memmaps), served by the
[Topiary Stream runtime](https://github.com/jesusluque/topiary-stream) with a
gate-governed pool, a guaranteed quality floor and an optional elastic memory
governor.

**Serving caveat, stated plainly:** this artifact does NOT load with stock
`mlx_lm`. It requires the Stream runtime (pure Python over stock MLX kernels
— no custom Metal):

```bash
python src/serve.py --artifact jesusluque/qwen3.5-35b-topiary-stream \
    --pool-k 32 --governor
```

## Measured results (Apple M5 Pro, 24 GB)

| Metric | Value |
|---|---|
| 4-bit size / served peak | 19.5 GB (unservable) / **12.5–13.9 GB** |
| Decode speed | 44–47 tok/s sustained (indicative; long-uptime machine) |
| HumanEval / GSM8K (greedy) | **92%** (23/25) / **92%** (46/50) |
| Exact-pager control (τ=0) | PPL 2.3614 vs true base 2.3623 — no measurable difference |
| Operating point τ=0.10 | +0.9% code / +0.6% wiki PPL |
| WikiText PPL vs best natively-fitting model | 7.11–7.83 vs 10.27 (−24 to −31%) |

## Honest limits

Pool policies pay in teacher-forced prefill (+6–11% PPL: flat prefill routing
defeats recency) while task decode is unaffected — the numbers above show
both. At n=25/50, task differences of ≤2 items are indistinguishable under
exact McNemar; treat the 92/92 as *indistinguishable from* the best fitting
model, not proven equal. Throughput figures predate a cold reboot.

## Provenance

Split deterministically from the community 4-bit conversion of
Qwen/Qwen3.5-35B-A3B with `split.py --layout resident-p0`; plane arithmetic
verified bit-exact against stock kernels in the repo's CI. Full measurement
history including negative results:
https://github.com/jesusluque/topiary-stream
