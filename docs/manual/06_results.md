# 6. Results compendium

All figures on an M5 Pro 24 GB, MLX, greedy, seed 1234. Each table names the
`runs/` file or the report it comes from. "n.s." = not significant at the
stated n.

## 6.1 The controls that anchor everything

| Control | Result | Source |
|---|---|---|
| P0+P1 == 4-bit through the kernel | error 2.6e-7 | `tests/test_stream.py` |
| Q8 = anchor + refinement vs true master | PPL 3.9209 vs 3.9213 | the lab, report 09 |
| Q2 derived by truncation vs native Q2 (OLMoE-7B) | ~20× worse PPL | the lab, report 09 |
| Pager τ=0 vs 35B base (TF-512) | 2.3614 vs 2.3623 (no measurable difference), peak 12.22 GB | README §Results |
| 80B exact mode (true base of a model that does not fit) | PPL 2.228 code / 5.557 general at 4.3 GB peak | paper §2.2 |
| Teacher-forced KLD with exact prefill | 0.000 over 2278 tokens | `runs/kld_k80served.json` |
| Centroid fold: real mean of the discarded plane | 1.504 (theoretical 1.5) | paper §1.1 |

## 6.2 The flagships: 35B and 80B

| | 35B (1.1× RAM) | 80B (1.8× RAM) |
|---|---|---|
| 4-bit size | 19.5 GB (Metal OOM on generation) | 42 GB |
| Served peak | 12.5–14.6 GB | 16.5–17.9 GB |
| tok/s | 44.5 (K=64) · 47.6 (K=32) · 47.2 cold, citable | 17.3–21.5 warm; 20.1 typical |
| Served PPL (exact prefill) | 2.3614 / 7.0583 = base | 2.2280 / 5.5569 = base |
| HumanEval / GSM8K | 92% / 92% (n=25/50, pool-prefill) · 14/15 / 15/15 (exact, n=15) | 15/15 / 15/15 (n=15; before exact prefill 84%/96%) |
| MATH-500 / MBPP (n=100) | 60 / 78 | 64–65 / 81 |
| MMLU / LAMBADA (n=500) | 83.0 / 75.4 | 85.2 / 78.2 |

Sources: `runs/bench_b35_*.json`, `runs/bench_b80_*.json`,
`runs/bench_c80best_*.json`, `runs/tasks_*.json`.

## 6.3 The solutions bench: five models, four axes

| Bench | 30B original | 30B Topiary | 30B-Stream | 35B Stream | 80B Stream |
|---|---|---|---|---|---|
| Served RAM | 16.4–17.9 GB | 14.5–15.2 GB | **9.2–11.3 GB** | 12.5–14.6 GB | 16.5–17.9 GB |
| MATH-500 (100) | 70 | **72** | 67 | 60 | 64 |
| MBPP (100) | **83** | 81 | 82 | 78 | 81 |
| MMLU (500) | 78.2 | 68.2 | 69.4 | 83.0 | **85.2** |
| LAMBADA (500) | 64.6 | 60.2 | 60.0 | 75.4 | **78.2** |

Controlled pairs: original/taper (same model): the taper preserves reasoning
(n.s.) and costs **−10 MMLU** and −4.4 LAMBADA; +6% tok/s and −2.7 GB.
Taper/30B-Stream (same checkpoint): indistinguishable on 3 of 4 axes with
5.3 GB less. The remaining columns mix families (directional only).
Source: `runs/bench_b30*.json`, `reports/bench_soluciones_20260819.md` §1.

## 6.4 Speed (256 tokens, warm machine)

| Config | tok/s | Peak GB |
|---|---|---|
| 30B original native | 101.4–103.0 | 17.2 |
| 30B Topiary native | 108.1–108.3 | 14.5 |
| 30B-Stream K=32 | 47.9 | 11.1 |
| 35B K=4 / K=32 / K=64 | 54.8 / 47.6 / 44.5 | 12.9 / 13.9 / 15.0 |
| 35B refresh=64 | 46.4 | 13.9 |
| 35B + governor | 51.1 | 13.9 |
| 80B C=240 r256 / r128 / r64 / r32 | 17.1–17.3 / 15.4 / 12.3 / 8.5–9.1 | 17.0 |
| 80B C=120 | 28.1 | 10.2 |
| 80B C=290 K=1 | ~17 | 19.8 |
| 80B + overflow tier r32 | 5.8 | — |
| Rival UD-Q2_K_XL (CPU) | 12.6 ± 3.7 | 30 GB paged |

Attribution (35B): compiling the graph changes nothing (30.8 vs 30.4); serving
floor-only (half the kernels and bytes) gains 3%: the pool costs ~3% and the
rest is the architecture's intrinsic batch-1 cost (hybrid SSM + 248k head).

## 6.5 Decode regime: KLD against the exact base (wiki 4×512, 16-token prefix)

| Config | Coverage | Mean KLD | p95 | p99 | Curve 0–64 → 256–448 | Source |
|---|---|---|---|---|---|---|
| 80B C=120 drops | 23% | 1.563 | — | 10.7 | 2.09 → 1.26 | `kld_k80drop120_long` |
| 80B C=120 + 25% floor | 23% (+floor) | 1.354 | — | 9.5 | — | `kld_k80floor120_long` |
| 80B C=240 K=32 r256 (production) | 47% | 0.774 | 4.40 | 9.02 | 0.88 → 0.94 → 0.79 → 0.68 | `kld_k80served_long` |
| 80B C=240 r128 | 47% | 0.566 | 3.11 | 7.56 | — | `kld_cand_r128` |
| 80B C=240 r64 | 47% | 0.416 | 2.01 | 5.55 | 0.62 → 0.38 | `kld_k80c240_r64` |
| 80B C=240 r32 | 47% | 0.303 | 1.37 | 4.29 | stationary 0.18 | `kld_cand_r32` |
| 80B C=290 all P0 | 57% | 0.582 | — | 7.5 | 0.63 → 0.51 | `kld_k80c340_long` |
| 80B absorb | 47% | 7.17 | — | — | flat ~7 | `kld_k80absorb_long` |
| 80B overflow tier r32 | 47% | 0.517 | — | — | — | `kld_k80ovf_r32` |
| 30B-Stream K=32, P0 for all | 100% P0 | **0.131** | 0.52 | 2.26 | 0.23 → 0.12 → 0.13 → 0.11 | `kld_k30served_long` |
| **Rival UD-Q2_K_XL** (top-100, CPU) | static | **0.195** | 0.91 | 1.94 | 0.27 → 0.22 → 0.15 → 0.18 | `kld_udq2_vs_base` |

Others: LAMBADA in decode (80 tokens, 16-token prefix) 80B: mean 0.118, p99
1.83. Greedy trajectories 300@32 (80B C=240 vs base): exact-match 36%, mean
divergence token 19.9 (Unsloth reports ≈25% for its UD-Q2_K_XL vs BF16).

Readings: coverage is the first-order term (half the pool doubles the
damage); the curves **decrease** with position (start-up damage, not
accumulation in the KV); cadence is the second-order term and its cost is
speed; the universal floor turns catastrophe into a moderate toll.

## 6.6 The duel against Unsloth UD-Q2_K_XL (same 80B, same machine)

| | 80B Topiary Stream (C=240, r128) | Unsloth UD-Q2_K_XL + llama.cpp |
|---|---|---|
| RAM | **17 GB** | 30 GB (pages from disk) |
| tok/s | **15.4–21.5** (GPU) | 12.6 ± 3.7 (CPU) |
| MATH-500 (100) | 65 | **69** |
| MBPP (100) | 81 | **86** |
| MMLU (500) | 85.2 | 86.2 (+5 items, n.s.) |
| Prose KLD vs base (wiki) | 0.774 (r256) · 0.303 (r32) | **0.195** |
| Prompt | exact | 2-bit |

Source: `runs/bench_udq2_*.json`, `runs/kld_udq2_vs_base.json`. Verdict in
§1.3.

Sources: rival artifact https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF ; methodology and the
published KLD yardstick (Gemma-27B vs BF16: Q2_K_XL 0.221, Q4_K_XL 0.024 — a
different model and reference, so not directly comparable with our column)
https://unsloth.ai/docs/basics/dynamic-3.0-ggufs ; KLD/flips rationale: *Accuracy is Not All You Need*
(arXiv:2407.09141).

## 6.7 Coverage on tasks (80B, same checkpoint)

| Config | Coverage | MATH-500 | MBPP | MMLU | LAMBADA | tok/s | Peak |
|---|---|---|---|---|---|---|---|
| C=120 | 23% | 44 | 72 | 86.0 | 78.2 | 28.1 | 10.2–11.8 |
| C=240 K=32 (production) | 47% | 64–65 | 81 | 85.2 | 78.2 | 15.4–17.3 | 17.9 |
| C=290 K=1 (2-bit mode) | 57% | 52 | 70 | — | — | ~17 | 19.8 |

Reading: half the pool breaks long decode (−20 MATH, −9 MBPP) without
touching knowledge (MMLU/LAMBADA intact); spending the detail on coverage
(C=290) also breaks the generative tasks (−13/−11). Source: `bench_b80_c120_*`,
`bench_c80c290_hard`.

## 6.8 Salient P1 subsampling (30B-Stream, n=100)

| Config (g:u:d) | Pool bytes | MATH-500 | MBPP | MMLU | Peak |
|---|---|---|---|---|---|
| 4:4:4 (default) | 3.0× | 67 | 82 | 69.4 | 11.3 |
| 4:4:2 | 2.5× | 72 | — | — | 12.1 |
| 2:2:4 | 2.0× | 69 | 74 | 65.8 | 12.0 |
| 2:4:2 | 2.0× | 65 | — | — | 12.1 |
| 2:2:2 (K=64) | 3.0× | 66 | — | — | 13.8 |
| 2:2:2 | 1.5× | 65 | — | — | 12.0 |
| 1:1:4 | ~1.2× | 64 | — | — | 12.0 |
| 0:0:4 | ~1.0× | 64 | — | — | 10.4 |

K dial on the 30B-Stream: K=4 65/75, MMLU 66.4 (10.6 GB); K=64 69/84, MMLU
69.2 (13.8 GB). Source: `runs/bench_b30s_*.json`.

## 6.9 The 235B (5.2× RAM): the negative that maps the frontier

| Mode | Result |
|---|---|
| drop-renormalize | 12 tok/s, collapsed output |
| universal 16.7% floor (floor256) | degenerate text at 5.7 tok/s (flat salience: 53.5% captured) |
| blocking floor | perfect text at 0.2 tok/s (94 syncs/token) |
| exact | batch-only |
| token-level retry-on-miss | falsified the same day: 99% retries (real condition `L·k·P(miss) ≪ 1`) |
| routing persistence (f48) | W8 77.7%, W16 86.8% (the tracking pool is viable on indices; without a universal floor the zeros destroy) |

Three converging walls: coverage 11–16% ≪ working set of 30–60
experts/layer; floor budget below the width cliff; flat salience. Falsifiable
prediction: with 48–64 GB (coverage ≈50%) this same stack serves it. Source:
the lab `runs/f44_*`, `f45_*`, `f47_*`, `f48_235b.log`.

## 6.10 Governor

External 8 GB balloon on the 35B: four automatic shrink steps (K 32→4,
active memory 13.7→12.8 GB), generation completed at 49.3 tok/s. Source:
the lab `runs/f46_balloon*.log`.
