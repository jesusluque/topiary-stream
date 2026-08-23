# 5. Evaluation protocol

## 5.1 Determinism

- **Greedy everywhere**: `make_sampler(temp=0.0)` in the tasks and
  trajectories; `temperature 0` in the API calls to the rival (in llama.cpp
  it is explicitly greedy; `repeat_penalty` defaults to 1.0 = off). No top-p,
  no top-k, no sampling anywhere.
- **Seeds**: `set_seeds(1234)`; under greedy they only affect which items
  enter each sample (`np.random.default_rng(1234).permutation`).
- **Thinking disabled** (`enable_thinking=False` in the chat template); the
  rival is the Instruct checkpoint (no thinking mode).
- Consequence: every figure is repeatable; the only source of variation
  between runs is the pool state, which depends on the item order (fixed).
- Same machine, same prompts, same parsers for every configuration and for
  the rival.

## 5.2 `eval_stream.py` — stages

```
python src/eval_stream.py --artifact <art> --stage <stage> [--serve-mode ...]
    [--pool-c C --pool-k K --orders ...] [--gen-refresh N] [--tag T] ...
```

| Stage | What it measures | Output |
|---|---|---|
| `ppl` | Teacher-forced PPL per serving mode (code / general; `--chunks-* --chunk-len`) | `runs/ppl_<tag>.json` |
| `tasks` | HumanEval (`openai/openai_humaneval`) and GSM8K (`openai/gsm8k`) via `datasets`; `--humaneval N --gsm N` | `runs/tasks_<tag>.json` |
| `bench` | MATH-500, MMLU, MBPP, LAMBADA from frozen `data/`; `--bench list --n N`; saves `per_item` | `runs/bench_<tag>.json` |
| `kld` | KL(base ‖ served) per token. Two passes: `--kld-out` saves the reference log-probs (`exact` mode); `--kld-ref` compares. `--kld-decode` for the decode regime; `--kld-refresh N` | `runs/kld_<tag>.json`, `runs/kld_<tag>_curve.npz` |
| `traj` | Greedy trajectory divergence (Divergence-300@32 style): same prompts, 32 tokens, at which token do they diverge? | `runs/traj_<tag>.json` |
| `kldremote` | KLD of an external baseline (llama-server `/completion`, `n_probs`) against **our** saved reference; KL truncated to the reference's top-100 | `runs/kld_<tag>.json` |

With `--openai-base URL` the tasks are sent to an OpenAI-compatible server
(the rival) with the same prompts and parsers; LAMBADA is skipped (it needs
teacher-forcing).

## 5.3 Frozen datasets (`data/`)

| File | Items | Sampling in `bench` | Verification |
|---|---|---|---|
| `math500.jsonl` | 500 | stratified by level, `n` | symbolic `math_verify` on the last `\boxed{}` (brace-balancing parser) |
| `mbpp.jsonl` | 257 (sanitized) | permutation, `n` | execution with `test_list` + `test_imports`, 15 s timeout, `PASS` |
| `mmlu500.jsonl` | 500 | first 500 (stratified by subject at freeze time) | letter `[ABCD]` in the answer (`max_tokens 8`) |
| `lambada.jsonl` | 5153 | permutation, `n` | teacher-forced: argmax of every token of the last word |
| `ifeval.jsonl` | 541 | — | unused |

Prompts: MATH "Solve step by step and put your final answer inside
\boxed{}" (1024 tokens); MBPP "Your function must satisfy this test: …
Reply with ONLY the complete Python function inside a ```python code block"
(512); HumanEval analogous (512); GSM8K "Reason briefly and end with the final
numeric answer" (384, last number).

**Intra-generation refresh**: `_ask` calls `refresh_all()` every
`GEN_REFRESH` tokens (and in the burst if enabled) — a harness finding: the
first batteries ran with the pool frozen throughout each answer
(~500 tokens). The production figures use `--gen-refresh 128`.

## 5.4 KLD: three regimes

1. **Teacher-forced in a single forward (T>1)**: measures the exactness of
   the prefill. With exact prefill it is 0.000 by construction (2278 tokens) —
   bit-for-bit identity. It does not measure the pool.
2. **Decode regime** (`--kld-decode`): exact prefix of `k0 = 16` tokens with
   a KV cache (`make_prompt_cache`), `refresh_all()` after the prefix, and
   the rest **token by token** feeding the true token (teacher-forced, but
   with the pool state evolving as in decode), refresh every `--kld-refresh`.
   It saves the per-position vector and reports mean/p95/p99/max and the mean
   per span 0–64 / 64–128 / 128–256 / 256–448 (counted from the prefix).
   **This is the KLD of the pool's toll.**
3. **Remote rival** (`kldremote`): same tokens via `/completion` with
   `cache_prompt` (each step evaluates one token: 1976 positions in 2.5 min
   on CPU), `n_probs 100`, pre-sampling probabilities (`top_logprobs`). KL
   over the reference's top-100 renormalized on both sides: a standard
   approximation that **underestimates** relative to the full KL; it is
   declared. Positions where llama-server returns no probabilities (partial
   UTF-8 token at the prompt boundary → response without
   `completion_probabilities` or HTTP 500) are skipped and counted
   (`skipped_eos`: 4 of 1980).

Saved references: `runs/kld80_base_long.npz` (80B exact base, wiki 4×512,
495 positions per chunk), `runs/kld30_base*.npz` (30B-Stream vs its taper).
Corpus: `nanite-moe/data/calib_general_qwen3/held_out.jsonl` (held-out,
never used for `orders`).

## 5.5 Flips and statistics

- `per_item` per benchmark in every `bench_*.json`; `flips(tag_a, tag_b)`
  counts correct→incorrect and incorrect→correct and computes exact McNemar
  (*Accuracy is Not All You Need*, 2407.09141).
- Power: at n=100 the 95% CI is ±9.4 points around 65%; at n=300, ±5.4; at
  n=500, ±4.2. Paired (same items) with ~20% discordant: standard error of
  the difference ≈ 2 points at n=500 → a 5-point gap comes out significant;
  at n=300 it sits on the edge. MBPP is capped at 257.
- The PPL equalities (four decimals over tens of thousands of tokens) are
  the statistically strong results; the tasks at n≤100 support
  "indistinguishable" and large effects (−20 from the coverage break), not
  fine rankings.

## 5.6 Speed

- `examples/speed_nativo.py`: one process per measurement (avoids Metal
  cache contamination), interleaved rounds, decode only (the first token does
  not count), 256 or 1024 tokens, median.
- `examples/citable_bench.sh`: after a **clean reboot**, 3 interleaved rounds
  of 1024 tokens; swap must be 0 (`sysctl vm.swapusage`) or the run is
  discarded. Citable: 35B 47.2 tok/s cold; 80B warm 17.3–21.5 (its cold
  rounds triggered 1.1 GB of swap and were discarded).
- Rival: `llama-bench`/`llama-server` with `-ngl 0` (the only mode that
  starts on 24 GB), `tg128`: 12.6 ± 3.7 tok/s.

## 5.7 The duel (protocol)

`examples/duel_udq2_v3.sh`: `llama-server -m <UD-Q2_K_XL.gguf> -ngl 0 -c 4096`,
waits for `/health`, smoke test with one chat request, and `eval_stream --stage bench
--openai-base http://127.0.0.1:8080` with the same `n` and seed. v1
(`-ngl 99`) died of a Metal OOM; v2 (`--cpu-moe`) died after loading. Run
in series with ours: its 30 GB paged plus our 17 GB do not fit together.

Rival: Unsloth Dynamic 3.0 GGUF, https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF (`-ngl 0`, b10520).

## 5.8 Tests (`pytest -q`, no models)

`test_unpack_pack_roundtrip`, `test_unpack_bits_order` (packing);
`test_plane_reconstruction_exact_through_kernel` (P0+P1 == 4-bit through
`gather_qmm`); `test_cold_centroid_bias` (1.5·s fold);
`test_anchored_q8_truncation_recovers_q4`; `test_pool_init_and_lookup`,
`test_pool_refresh_incremental_and_churn_cap`, `test_pool_p1_subset_invariant`
(pool state machine with synthetic memmaps; K ⊆ C always);
`test_governor_thresholds`; `test_model_card_placeholder_gate`. CI pinned to
the tested versions.
