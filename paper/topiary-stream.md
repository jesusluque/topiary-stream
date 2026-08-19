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
while improving general-domain perplexity by 31%; an 80B serves at 20–21.5
tok/s with served PPL equal to its true base and HumanEval/GSM8K 15/15
(n=15) under exact prefill; a 235B is shown to have *no servable middle ground* at this
memory — a negative we map to three converging, measured walls. A
four-benchmark suite (MATH-500, MBPP, MMLU, LAMBADA) over five servable
configurations separates what each compression axis costs: static salience
pruning preserves reasoning but cuts knowledge (−10 MMLU points against its
own unpruned base); paged residency preserves both at the cost of speed; and
an intra-model coverage ablation (same checkpoint, pool halved) breaks
long-form reasoning by 20 points while leaving short-form knowledge intact —
turning the coverage law from a cross-model observation into a controlled,
same-model result.

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

### 1.4 Exact prefill and partial-detail variants

Prompt processing is one batched pass, so it is served exact by design: for
any multi-token forward, P0 is read as usual and the P1 planes for the
prompt's expert *union* are gathered once from the memmaps. The pool policy
therefore governs decode only. This single design choice erased the
runtime's largest measured toll — pool-served prefill had cost +6–11% PPL on
the 35B and +28%/+120% on the 80B (flat prefill routing defeats
recency-based membership) — and lifted the 80B from 84%/96% to 15/15 / 15/15
on HumanEval/GSM8K.

Two floor refinements are implemented and measured (§2.6): `--p1-frac g,u,d`
serves P1 only for a salient-channel prefix per projection (a contiguous
slice, because Topiary checkpoints ship with channels salience-ordered), and
`--centroid empirical` folds the per-group empirical mean of the dropped
plane into the floor bias instead of the uniform 1.5s. The second is a
documented negative: the dropped plane is near-uniform noise.

### 1.5 The elastic governor

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
With exact prefill (§2.2) the fast-path's teacher-forced PPL equals the base
(2.3614 / 7.0583) and a task mirror scores 14/15 / 15/15 (n=15).

### 2.2 Scaling up: 80B (1.8×) and the true-base control

The 80B serves at 20–21.5 tok/s / 17.6 GB. Exact-mode evaluation — every slot
served P0+P1 straight from memmaps at 4.3 GB peak — measures the true base of
a model that cannot be loaded: PPL 2.228/5.557, the strongest base on this
hardware. Under pool-served prefill the teacher-forced toll was heavy in
dispersed domains (+28% code, +120% general) with a measured mechanism —
prefill routing is flat, so recency-based membership covers little — which
motivated the design fix: **prefill is served exact** (the prompt is one
batched pass; the expert-union's P1 reads once from the memmaps), confining
the pool policy to decode. With exact prefill, served PPL equals the true
base to four decimals and tasks reach HumanEval 15/15 / GSM8K 15/15 (n=15;
pool-prefill serving had measured 84%/96% — the prefill toll, not the decode
policy, was the handicap).

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

### 2.4 The solutions bench: what each compression axis costs

A single four-benchmark battery (MATH-500 n=100 level-stratified, MBPP
sanitized n=100, MMLU n=500 subject-stratified, LAMBADA n=500; greedy, fixed
seed, identical samples) over every servable configuration on this machine:

| | 30B original | 30B Topiary (pruned) | 30B-Stream | 35B Stream | 80B Stream |
|---|---|---|---|---|---|
| served RAM | 16.4–17.9 GB | 14.5–15.2 GB | **9.2–11.3 GB** | 12.5–14.6 GB | 16.5–17.9 GB |
| MATH-500 | 70% | **72%** | 67% | 60% | 64% |
| MBPP | **83%** | 81% | 82% | 78% | 81% |
| MMLU | 78.2% | 68.2% | 69.4% | 83.0% | **85.2%** |
| LAMBADA | 64.6% | 60.2% | 60.0% | 75.4% | **78.2%** |

Two controlled pairs sit inside this table. **Original vs. pruned** (same
model): salience pruning preserved reasoning entirely (MATH 72 vs 70, MBPP 81
vs 83 — both n.s.) but **cost 10 MMLU points** (68.2 vs 78.2; 50 items at
n=500) and 4.4 LAMBADA points. The taper removes *knowledge*, not
*reasoning* — invisible to reasoning-only batteries, which is how it went
unnoticed until this suite. Same-day interleaved decode-only rounds put the
speed dividend at +6% (108.1 vs 101.4 tok/s) with −2.7 GB. **Pruned vs.
Stream-served** (same checkpoint): the paged artifact is indistinguishable on
3 of 4 axes (MBPP +1, MMLU +1.2, LAMBADA −0.2; MATH −5 at the edge of
significance) while dropping peak memory from 14.5 to 9.2 GB — the champion
now fits 16 GB machines. The remaining columns mix model families
(Qwen3 / Qwen3.5 / Qwen3-Next), so their reading is directional: the large
paged models dominate exactly where total parameter mass lives (knowledge:
+15–18 points of MMLU/LAMBADA over the 30Bs), while the pruned 30B keeps the
reasoning crown.

### 2.5 The coverage law, third point: a same-model break

The 235B negative left the coverage frontier as two points confounded by
family, expert count and shared-expert presence (§2.3). Halving the 80B's
pool (C=240→120; 47%→23% of 512 experts/layer, same checkpoint, same
runtime) removes the confound: MATH-500 collapses 64%→44% and MBPP 81%→72%,
while short-form MMLU is intact at the C=120 operating point (90% on the
first 100 items, final pending). The mechanism is now isolated: under exact
prefill, dropping non-resident experts damages *long decode chains*
specifically — errors compound over hundreds of tokens — while single-token
answers survive. Coverage below the routed working set does not degrade
smoothly; it breaks, and it breaks reasoning first.

### 2.6 Salience-prefix subsampling and two measured negatives

With salience-ordered channels (free in Topiary checkpoints), P1 can be
served for only the top fraction of each projection's channels. An
eight-point matrix on the 30B-Stream (notation gate:up:down; 4 = full P1,
2 = 50% salient prefix, 0 = floor only):

| config | pool bytes | MATH-500 | MBPP | MMLU |
|---|---|---|---|---|
| 4:4:4 (default) | 3.0× | 67% | 82% | 69.4% |
| 4:4:2 | 2.5× | 72% | — | — |
| 2:2:4 | 2.0× | 69% | 74% | 65.8% |
| 2:4:2 | 2.0× | 65% | — | — |
| 2:2:2 (K=64) | 3.0× | 66% | — | — |
| 2:2:2 | 1.5× | 65% | — | — |
| 1:1:4 | ~1.2× | 64% | — | — |
| 0:0:4 | ~1.0× | 64% | — | — |

The MATH-vs-bytes curve is strikingly flat — three points from 3.0× down to
1.0× — but the full battery on 2:2:4 falsifies the free lunch: −8 MBPP and
−3.6 MMLU against the default. At n=100 the 64–72% MATH spread is not
pairwise-orderable; what is defensible is that the salient prefix is a
*memory-emergency dial* (0:0:4 serves usable MATH at 10.4 GB peak), not a
general optimization. Full P1 remains the default, and the recipe must be
revalidated per model — non-Topiary checkpoints do not ship salience-ordered
channels.

Two cheaper ideas were measured to death first. The *empirical centroid*
(per-group mean of the dropped plane folded into the floor bias, replacing
the uniform 1.5s) improves floor MSE by only 1.5%: the dropped low bits are
near-uniform noise (global mean 1.504 vs 1.5 theoretical). And *row-delta
compression* (1-bit change mask between rows plus changed values, motivated
by video codecs) was falsified directly: the fraction of identical values
between neighboring rows or neighboring experts equals the chance rate
exactly (8.6% measured vs 8.6% from the code distribution on q4), so the
scheme expands to 4.66 bits/weight. Quantized trained weights carry no
row-to-row redundancy to exploit; the exploitable structure is *which*
weights matter (salience), not *what values* they share.

## 3. Related work

SliceMoE [slicemoe2026] caches experts at bit-slice granularity with a
truncation-compatible Matryoshka quantization — hardware-oriented co-design
whose AMAT grid redesign independently corroborates our metric inversion.
PagedWeight [pagedweight2026] pages expert precision under a Hessian-aware
planner to free KV memory on datacenter GPUs. WiSP [wisp2026] casts
low-resource MoE serving as Denning working-set management with
byte-identical outputs; our working-set measurements instantiate that frame.
Cache-conditional routing [cacheconditional2025] biases the *router* toward
resident experts; our dynamic biases act on the *served representation*,
leaving routing untouched. On Apple Silicon, a 2026 wave of SSD-streaming
engines (flash-moe, SwiftLM, mlx-moe) serves oversized MoE at whole-expert
granularity with custom pipelines and no published quality accounting —
precisely the two gaps (sub-expert planes on stock kernels; measured floors
and tolls) this work fills; the lineage runs back to LLM-in-a-flash
[llmflash2024]. HOBBIT [hobbit2024], MoE-Infinity [moeinfinity2024],
ExpertFlow [expertflow2024], ProMoE [promoe2024], Fiddler [fiddler2025],
KTransformers [ktransformers2025] and MoE-Lightning [moelightning2025]
offload whole experts on CUDA/x86. Any-Precision LLM's bit-plane engine
[anyprecision2024] motivates our stock-kernel constraint; MatQuant
[matquant2025] co-trains the nested scales we anchor instead, and AWQ
[awq2024] states the salient-weights principle our truncation anatomy
(§1.1) reaches through the grid. The static companion is Topiary
[luque2026topiary]. Speculative decoding on hybrid SSM architectures is an
implementation gap in MLX today; The Mamba in the Llama [mambainllama2024]
gives the general solution.

**Concurrent work (post-July 2026, found in an adversarial sweep on
2026-08-19).** Tied Trit-Planes [tiedtritplanes2026] publishes a
plane-decomposition that folds losslessly into one 4-bit code plane and
disk-streams a 284B MoE on 64 GB consumer machines including Apple hardware —
through a bespoke Zig/Metal engine with LRU residency and no quality floor.
vLLM-Moet [vllmmoet2026] serves a 2-bit expert base with a confidence-gated
FP4 recovery tier on Blackwell GPUs via hand-written SASS. Both narrow the
novelty of "bit-planes for oversized MoE" as such; what remains unclaimed,
and what we therefore lead with, is the *conjunction*: planes each servable
by unmodified stock kernels, a hard miss-never-blocks floor, and quality
accounting (PPL = base, task parity) published with every policy. A
trace-driven-evaluation audit [traceeval2026] — run on MLX on a 24 GB M4
Pro, our exact hardware class — shows replay artifacts inverting expert-cache
policy rankings and mandates reporting the union-to-capacity ratio r̄; our
coverage experiments are live-served (no trace replay), and our working-set
measurements (~30–60 experts/layer against C) are the quality-linked
counterpart of that ratio.

Benchmarks: [math2021, math500-2023, mbpp2021,
mmlu2021, lambada2016, humaneval2021, gsm8k2021, wikitext2017]. Full BibTeX
in `paper/references.bib`; every identifier verified against the arXiv API
or publisher on 2026-08-19.

## 4. Limitations

Single machine, single model family (Qwen) plus OLMoE as a bench; task
n=25/50 (exact-McNemar indistinguishability radii reported); PPL anchors for
the 35B under a short protocol; throughput measured on a long-uptime machine;
the 235B artifact mixes DWQ with plain 4-bit siblings. Exact prefill removes
the pool's teacher-forced toll by construction; the residual decode-side toll
is bounded only by task results at small n. In the solutions bench, only the
original/pruned and pruned/Stream pairs are same-model controlled — the
cross-scale columns confound technique with model family. The P1-subsampling
matrix is n=100 per cell: its 64–72% MATH spread is not pairwise-orderable,
and the salient-prefix mechanism itself requires salience-ordered channels
(free only in Topiary checkpoints). The C=120 short-form results and the
K-dial quality batteries were in flight at writing time and are reported as
partial where marked.

## 5. Reproducibility and artifacts

Every number descends from committed configs and JSON run records
(`runs/*.json`, `reports/bench_soluciones_20260819.md`); plane arithmetic and
the pool state machine are covered by a pure-logic test suite run in CI on
Apple Silicon (macos-14).

- **Code**: github.com/jesusluque/topiary-stream — the eight-tool runtime
  (split/serve/pager/fastpath/pyramid/salience/floor/eval), tests, frozen
  benchmark datasets (`data/`), and this paper.
- **Servable artifacts** (Hugging Face, pending public release):
  `jesusluque/qwen3.5-35b-topiary-stream` (resident-P0, 18 GB),
  `jesusluque/qwen3-next-80b-topiary-stream` (full-memmap skeleton + planes,
  42 GB, includes routed-salience orders),
  `jesusluque/qwen3-235b-topiary-stream-kit` (skeleton + universal floor +
  orders + rebuild recipe; the 127 GB of planes rebuild in ~1 h with
  `split.py --consume`).
- **Static companion** (public): github.com/jesusluque/topiary and the
  pruned checkpoints `jesusluque/qwen3-30b-topiary` (the taper),
  `-w640`, `-w576-code`. The 30B-Stream artifact of §2.4 is produced from
  the taper checkpoint with `split.py --layout resident-p0` in ~15 min.

Serving is one command per artifact, e.g.
`python src/serve.py --artifact artifacts/qwen35-stream --pool-k 32 --governor`.
