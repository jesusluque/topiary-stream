---
license: apache-2.0
base_model: {BASE_MODEL}
library_name: mlx
tags:
- mlx
- moe
- topiary-stream
- quantized
- bit-planes
---

# {MODEL_NAME} — Topiary Stream artifact ({LAYOUT})

A servable residency artifact for a model that does not fit whole in
{RAM_GB} GB: every expert split into standard-kernel-readable bit planes
(P0 floor + P1 completion), served by the
[Topiary Stream runtime]({GITHUB_URL}) with a gate-governed pool and a
guaranteed quality floor.

**Serving caveat, stated plainly:** this artifact does NOT load with stock
`mlx_lm`. It requires the Stream runtime (`serve.py`, ~small pure-Python on
top of stock MLX kernels — no custom Metal):

```bash
python src/serve.py --artifact {HF_REPO} --pool-k {POOL_K} {EXTRA_FLAGS}
```

## Measured results (Apple M5 Pro, 24 GB)

| Metric | Value |
|---|---|
| 4-bit size / served peak | {SIZE_GB} GB / {PEAK_GB} GB |
| Decode speed | {TOKS} tok/s (indicative; long-uptime machine) |
| HumanEval / GSM8K (greedy, n={N_TASKS}) | {HUMANEVAL} / {GSM8K} |
| PPL code / general (mode {PPL_MODE}) | {PPL_CODE} / {PPL_WIKI} |
| True-base control (exact mode) | {BASE_PPL_CODE} / {BASE_PPL_WIKI} |

## Honest limits

{LIMITS_PARAGRAPH}

Task-n statistics: at n=25/50, differences of ≤2 items are indistinguishable
under exact McNemar — treat close scores as ties, not wins.

## Provenance

Split from `{BASE_MODEL}` with `split.py --layout {LAYOUT}` (deterministic;
byte-exact plane arithmetic verified against stock kernels in the repo's test
suite). Full measurement history and negative results: {GITHUB_URL}.
