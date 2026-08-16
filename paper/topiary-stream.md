# Topiary Stream: Gate-Governed Bit-Plane Residency for Serving Oversized MoE Models on Apple Silicon

**jesus luque** — draft, August 2026

## Abstract

We present a residency runtime that serves quantized MoE models larger than
available memory on consumer Apple Silicon, using only stock MLX quantized
kernels. Our contribution is a *conjunction* none of the concurrent systems
provide together: (i) an **anchored precision pyramid** whose levels are
directly consumable by standard affine-quantization kernels — no dedicated
engine; (ii) a **guaranteed quality floor as a design invariant** — a routed
expert is never served nothing, and every serving policy ships with its
measured toll; and (iii) **coverage laws with a falsifiable prediction**,
charted by serving three model scales (1.1×, 1.8×, 5.2× of RAM) to their
respective outcomes. On a 24 GB MacBook Pro, a Qwen3.5-35B whose 4-bit
checkpoint cannot generate at all serves at 44.5 tok/s within 14 GB, matching
the strongest natively-fitting model on HumanEval/GSM8K (92%/92%, n=25/50)
while improving general-domain perplexity by 31%; an 80B serves at 21.5 tok/s
with GSM8K 96%; a 235B is shown to have *no servable middle ground* at this
memory — a negative we map to three converging, measured walls.

## 1. Method

### 1.1 Anchored bit-plane pyramid

A 4-bit affine tensor with codes q4 splits exactly into two 2-bit planes,
q4 = 4·q_hi + q_lo, giving w = [4s]·q_hi + [s]·q_lo + β. Each plane is a
*valid affine tensor* for MLX's `gather_qmm`: level-4 reads P0+P1 (bit-exact,
verified to 2.6e-7 through the kernel); the floor reads P0 alone with a
centroid-folded bias (β+1.5s).

Construction direction matters and we measure why. Deriving low levels by
truncating a fine master *wins* on uniform weight metrics — L2 AND max error —
yet *loses* end-to-end (derived Q2: ~20× worse PPL than a native Q2 fit on
OLMoE-7B). The truncated grid concentrates its error on salient weights,
which uniform metrics under-weight; this is the mechanistic face of the
salient-weights principle (AWQ) arrived at through grid truncation, and the
reason Any-Precision LLM upscaled from the bottom, MatQuant co-trains scales,
and SliceMoE redesigns its grid (AMAT). The corrected rule: **anchor the
pyramid at the serving level** — Q4 as a native fit, Q8 as anchor+refinement
(measured free: PPL 3.9209 vs true-master 3.9213), Q2 as anchor-truncated,
valid *only* for gate-protected cold slots.

### 1.2 Artifacts: resident-P0 and full-memmap

`split.py` turns a community 4-bit checkpoint into a self-contained servable
artifact. For models whose P0 fits (≤ ~60% of the 4-bit size), P0 stays in
the checkpoint's weight slots and P1 goes to per-(layer, projection) row-
pageable memmaps. Beyond that, nothing expert-shaped stays resident: P0, P1
and scales/biases all page by expert row, and a 3–5 GB *skeleton* checkpoint
(non-expert weights plus 1×64 dummy switch tensors restored by a constructor
patch) makes the model loadable without its 40–122 GB of experts.

### 1.3 Gate-governed pool, sync-free

The router's scores exist before any expert bytes are read — residency
decisions are free. A per-layer pool holds K experts' P1 (resident-P0 layout)
or C experts' P0 plus K of them with P1 (full-memmap). Membership is encoded
in a *dynamic biases tensor* — pool members carry β, others β+1.5s — so one
kernel call serves hot and cold with no per-token CPU/GPU synchronization.
Routing counts drain into a frequency+recency EMA at one deferred sync per
128–256 tokens; refreshes are incremental (a full pool rebuild costs 10× in
decode — MLX `setitem` copies whole buffers). Misses never block: the slot
serves its floor this token and the pool learns.

Sync elimination is the speed story: the same 35B artifact serves at 8.1
tok/s with per-layer synchronization and 44.5 tok/s without. Attribution
experiments bound remaining overheads: graph compilation changes nothing
(30.8 vs 30.4 — Python was never the bottleneck), and floor-only serving
(half the expert kernels and bytes) gains 3% — the pool costs ~3% and the
rest is the architecture's intrinsic batch-1 cost (hybrid SSM layers, a
248k-vocab head).

### 1.4 The elastic governor

Residency is a dial, so we attach a controller: each refresh reads macOS's
real available memory and resizes K stepwise between bounds. Measured under
an adversarial 8 GB allocation: four automatic shrink steps (K 32→4, active
memory tracking down 13.7→12.8 GB) with generation completing cleanly.

## 2. Results

### 2.1 The flagship: serving the unservable (35B, 1.1× RAM)

Qwen3.5-35B-A3B 4-bit loads at 19.5 GB and dies on generation on a 24 GB
machine. Served through the pager: the τ=0 control shows **no measurable
difference** from the base (PPL 2.3614 vs 2.3623, protocol TF-512) at 12.22 GB
peak — reproduced across independent processes to four decimals. The τ=0.10
operating point costs +0.9%/+0.6% PPL. Through the full fast-path (pool K=32),
HumanEval 92% (23/25) and GSM8K 92% (46/50) — indistinguishable from the best
natively-fitting model (92%/94%) — with WikiText PPL 7.11–7.83 vs its 10.27.

### 2.2 Scaling up: 80B (1.8×) and the true-base control

The 80B serves at 21.5 tok/s / 17 GB with GSM8K 96%. Exact-mode evaluation —
every slot served P0+P1 straight from memmaps at 4.3 GB peak — measures the
true base of a model that cannot be loaded: PPL 2.228/5.557, the strongest
base on this hardware. The pool's teacher-forced toll is heavy in dispersed
domains (+28% code, +120% general) while task decode is unaffected: under
this runtime the 80B is a reasoning specialist. Toll asymmetry has a measured
mechanism: prefill routing is flat (recency-based membership covers little);
decode routing is local.

### 2.3 The negative that maps the boundary: 235B (5.2×)

Four serving modes measured: drop-renormalize runs at 12 tok/s but output
collapses (naked drops of dominant slots destroy, not degrade); a universal
16.7%-width salience floor produces degenerate output (the model's per-expert
salience is flat: the prefix captures 53.5% vs ~94% on concentrated models);
a blocking floor yields *perfect* text at 0.2 tok/s (94 sync points/token);
exact mode works batch-only. Three converging walls: pool coverage (11–16% of
experts ≪ the observed ~30–60-experts/layer working set), floor budget below
the width cliff, flat salience. Token-level retry-on-miss was predicted to
converge at C ≫ working set and **falsified same-day** (99% retries): the
correct condition is L·k·P(miss) ≪ 1, unreachable against load-balanced
expert tails. Falsifiable prediction: at 48–64 GB (coverage ≈50%) this stack
serves the 235B with nothing new built.

## 3. Related work

SliceMoE (DAC'26) caches experts at bit-slice granularity with a
truncation-compatible Matryoshka quantization — hardware-oriented co-design
whose AMAT grid redesign independently corroborates our metric inversion.
PagedWeight (2026) pages expert precision under a Hessian-aware planner to
free KV memory on datacenter GPUs. WiSP (2026) casts low-resource MoE serving
as Denning working-set management with byte-identical outputs; our working-set
measurements instantiate that frame. Cache-conditional routing (Skliar et
al.) biases the *router* toward resident experts; our dynamic biases act on
the *served representation*, leaving routing untouched. On Apple Silicon, a
2026 wave of SSD-streaming engines (flash-moe, SwiftLM, mlx-moe) serves
oversized MoE at whole-expert granularity with custom pipelines and no
published quality accounting — precisely the two gaps (sub-expert planes on
stock kernels; measured floors and tolls) this work fills. HOBBIT, MoE-
Infinity, ExpertFlow, ProMoE, Fiddler, KTransformers and MoE-Lightning
offload whole experts on CUDA/x86. Any-Precision LLM's bit-plane engine
motivates our stock-kernel constraint. Speculative decoding on hybrid SSM
architectures is an implementation gap in MLX today; The Mamba in the Llama
gives the general solution.

## 4. Limitations

Single machine, single model family (Qwen) plus OLMoE as a bench; task
n=25/50 (exact-McNemar indistinguishability radii reported); PPL anchors for
the 35B under a short protocol; throughput measured on a long-uptime machine;
the 235B artifact mixes DWQ with plain 4-bit siblings. The pool's prefill
toll makes teacher-forced perplexity a hostile metric for this runtime —
we report it anyway.

## 5. Reproducibility

Every number descends from committed configs and JSON run records; plane
arithmetic and the pool state machine are covered by a pure-logic test suite
run in CI on Apple Silicon. Artifacts (35B, 80B) and a 235B build kit are
published with the runtime.
