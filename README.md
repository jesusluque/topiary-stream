# Topiary Stream

*Serve MoE models that don't fit.*

> Topiary shapes checkpoints like a gardener — its sibling, Stream, keeps the
> whole tree alive on disk and lets your RAM hold only the branches the wind
> is actually moving.

Topiary Stream is a residency runtime for quantized MoE models on Apple
Silicon (MLX): an **anchored bit-plane pyramid** (every expert splits into
standard-kernel-readable precision planes), a **gate-governed pool** (the
router score is known *before* the expert weights are read — residency
decisions are free), a **guaranteed quality floor** (a miss never blocks and
never serves garbage), and an **elastic governor** that resizes residency
against real memory pressure, live.

What that buys on one MacBook Pro (M5 Pro, 24 GB), all measured:

```
model                     4-bit size   served peak   HumanEval  GSM8K   tok/s
Qwen3.5-35B-A3B             19.5 GB*     12.5-14 GB      92%     92%    44.5
Qwen3-Next-80B-A3B          42   GB      17.6  GB     15/15    15/15    20.1
Qwen3-235B-A22B            132   GB      — no servable middle exists (see limits)
    * loads, but Metal OOMs on generation — the model cannot be served whole
    (80B task n=15 under exact-prefill serving; served PPL == its true base)
```

The 35B row matches the best model that *does* fit this machine (a tuned
Qwen3-30B: 92% / 94%) while beating it on general knowledge (WikiText PPL
7.11 vs 10.27) — from a model the machine cannot natively run. Everything is
served by **stock MLX quantized kernels**: no custom Metal, no dedicated
inference engine.

## The idea, in three steps

1. **Split every expert into bit planes.** A 4-bit affine tensor is exactly
   two 2-bit tensors (`q4 = 4·q_hi + q_lo`), each valid for `gather_qmm` as
   shipped. P0 (high bits, with a centroid bias fold) is a servable floor at
   half the bytes; P0+P1 is bit-exact 4-bit. The pyramid is *anchored* at the
   serving level — refining upward is measured-free, truncating downward is
   only sound under gate protection (deriving levels top-down wins on weight
   L2 and max error yet loses ~20× end-to-end; the error lands on salient
   weights).
2. **Let the gate govern residency.** Expert routing scores exist before any
   expert bytes are read. A per-layer pool keeps the hot experts' planes
   on-GPU, membership encoded in a dynamic biases tensor so ONE kernel call
   serves hot and cold — no per-token CPU sync (that sync was the difference
   between 8.1 and 44.5 tok/s). A miss serves the floor ("blurry for one
   frame") and the pool learns it at the next deferred refresh.
3. **Breathe.** The governor reads real memory pressure (vm_stat) each
   refresh and resizes the pool. Measured: four automatic shrink steps under
   an 8 GB external allocation, generation completing cleanly throughout.

## Quickstart

```bash
git clone https://github.com/jesusluque/topiary-stream && cd topiary-stream
uv venv && source .venv/bin/activate && uv pip install -e .

# 1. Split a community checkpoint into a servable artifact (~15 min)
python src/split.py --src mlx-community/Qwen3.5-35B-A3B-4bit \
    --out artifacts/qwen35-stream --layout resident-p0

# 2. Serve it (a model this machine cannot load whole)
python src/serve.py --artifact artifacts/qwen35-stream --pool-k 32 --governor

# 3. Optional: watch it breathe under pressure
python examples/balloon.py   # in a second terminal
```

For models whose P0 alone exceeds RAM, use `--layout full-memmap` (skeleton +
row-pageable planes) and size the pool with `--pool-c`.

## Tools

| Tool | What it does |
|---|---|
| `split.py` | checkpoint → servable artifact (two layouts; `--consume` caps peak disk) |
| `serve.py` | generation CLI: fast-path or unified pool, warm-up, governor, stats |
| `fastpath.py` | resident-P0 runtime: sync-free decode, dynamic biases, elastic governor |
| `pager.py` | full-memmap runtime: unified pool, serving modes exact/nosync/floor/floor2d |
| `pyramid.py` | anchored Q2/Q4/Q8 pyramid from an 8-bit master (plus the negative-control builders) |
| `salience.py` | routed-salience profiling **through the pager** — no checkpoint needed |
| `floor.py` | universal 2D floor (P0 × salience prefix) for `floor2d` |
| `eval_stream.py` | the quality accounting: PPL per serving mode + HumanEval/GSM8K |

## Results

**The flagship (Qwen3.5-35B-A3B — cannot be served whole on 24 GB):**

| Config | Code PPL | Wiki PPL | Peak GB |
|---|---|---|---|
| true 4-bit base (control) | 2.3623 | 7.0716 | 20.3 (crashes on generation) |
| pager, τ=0 control | 2.3614 — no measurable difference | — | **12.22** |
| pager, τ=0.10 operating point | 2.3830 (+0.9%) | 7.1114 (+0.6%) | 12.22 |
| fast-path tasks, pool-served prefill (superseded) | HumanEval **92%** · GSM8K **92%** (n=25/50) | | 14.4 |
| **fast-path + exact prefill — the shipping default** | HumanEval **14/15** · GSM8K **15/15** (n=15) · TF PPL 2.3614/7.0583 = base | | 14.05 |

**The 80B (Qwen3-Next-80B-A3B, 1.8× RAM):** with exact prefill, served PPL
equals its true base to four decimals (2.2280 / 5.5569 — the strongest base
this machine has touched) at 20–21.5 tok/s. Tasks under the full runtime:
HumanEval **15/15**, GSM8K **15/15** (n=15; earlier pool-prefill serving had
measured 84%/96% — the prefill toll was the whole handicap).

**Speed attribution** (why we trust 44.5 tok/s is near the ceiling): compiling
the block graph changed nothing (30.8 vs 30.4 — Python was not the
bottleneck); serving floor-only (half the expert kernels and bytes) gained 3%
(31.7) — the pool costs ~3% and ~31-45 tok/s is the *architecture's* intrinsic
speed here (hybrid mamba at batch-1 + a 248k-vocab head).

## Honest limits (all measured, not speculated)

- **Prefill is served exact by design** (the prompt is one batched pass; P1
  for its expert-union reads once from the memmaps), so the pool policy
  governs decode only. Before this design the pool's prefill toll measured
  +6–11% PPL on the 35B and +28%/+120% on the 80B — flat prefill routing
  defeats recency — which is why the split exists. Decode-side pool cost is
  bounded by the task results above.
- **The 235B has no servable middle on 24 GB.** Four modes measured:
  drop-renormalize = fast but collapsed output; universal 16.7%-width floor =
  degenerate (salience too flat: 53.5% captured); blocking floor = perfect
  text at 0.2 tok/s; exact = batch-only. Three converging walls: pool
  coverage (11–16% ≪ the ~30–60 experts/layer working set), floor budget
  below the width cliff, flat salience. Falsifiable prediction: with 48–64 GB
  (pool ≈50%) the same stack serves it — nothing new to build.
- **Coverage laws.** Every routed slot needs at least a floor — naked drops
  of dominant slots collapse the model, not degrade it. Token-level
  retry-on-miss requires `L·k·P(miss) ≪ 1` and is unreachable in deep models
  with load-balanced tails (falsified live: 99% retries at C ≫ working set).
- **Speculative decoding is blocked on Qwen's hybrid generation** in current
  mlx-lm (DeltaNet/mamba state caches are not trimmable). This is an
  implementation limit, not a fundamental one — see *The Mamba in the Llama*
  (NeurIPS 2024) for the general solution; vLLM supports it partially.
- **Statistics.** Task n=25/50: differences of ≤2 items are indistinguishable
  (exact McNemar), so "92% vs 92%" means *indistinguishable*, not proven
  equal. tok/s figures were measured on a long-running machine (indicative);
  cold-boot numbers pending.
- **Cold storage must stay off the hot path.** Paging the floor from a USB
  disk was >4× slower at minimum (the run had to be abandoned unfinished).
  Slow tiers are for refresh-time reads only.

## Repository map

```
src/          the eight tools above + common.py (plane arithmetic, discovery)
tests/        pure-logic suite (planes, pool state machine, governor) — CI on macos-14
examples/     balloon.py (governor pressure demo)
paper/        write-up draft
huggingface/  model-card template + hardened upload script
```

## Relation to Topiary

[Topiary](https://github.com/jesusluque/topiary) is the static half: shape a
checkpoint once by routed salience, serve it anywhere. Stream is the dynamic
half: keep everything, page by gate, choose quality per token. They compose —
a checkpoint can be prefix-servable *and* plane-servable (the width and bits
axes stack sub-additively).

## Citing

```bibtex
@misc{luque2026topiarystream,
  title  = {Topiary Stream: Gate-Governed Bit-Plane Residency for Serving
            Oversized MoE Models on Apple Silicon},
  author = {luque, jesus},
  year   = {2026},
  url    = {https://github.com/jesusluque/topiary-stream}
}
```

## License

MIT.
