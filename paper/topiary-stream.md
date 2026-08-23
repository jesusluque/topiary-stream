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
same-model result. Finally we submit the runtime to hostile metrics —
token-by-token decode KLD with a live cache, greedy-trajectory divergence —
and to a head-to-head duel against a calibrated static 2-bit of the same 80B
(Unsloth UD-Q2_K_XL): task quality is at par on knowledge and ~5 points
behind on generative tasks, while Stream fits in 17 GB instead of 30, runs
40–70% faster and serves the prompt exactly. Four cheaper "fixes" for the
generative toll were measured and retired; the honest residue is the quality
of the 2-bit plane itself.

## 1. Method

### 1.1 Anchored bit-plane pyramid

A 4-bit affine tensor with codes q4 splits exactly into two 2-bit planes,
q4 = 4·q_hi + q_lo, giving w = [4s]·q_hi + [s]·q_lo + β. Each plane is a
*valid affine tensor* for MLX's `gather_qmm`: level-4 reads P0+P1 (exact up to
float accumulation order — 2.6e-7 through the kernel — and teacher-forced
KLD 0.000 over 2,278 tokens, §2.7); the floor reads P0 alone with a
centroid-folded bias (β+1.5s). A scoping note: the exact plane split requires
*flat affine* quantization (w = s·q + β per group). It does not port to
GGUF's strong formats — K-quants nest 6-bit scales inside superblocks and
IQ-quants are codebook-based — so the pyramid lives in the MLX affine world
(of GGUF types, only Q4_1-style block affine is structurally compatible).
This is the flip side of the stock-kernel property, and we state it rather
than imply generality. The 1.5s fold is the expectation-correct
(first-order unbiased) constant under uniform q_lo; we measured the dropped
plane to be near-uniform in practice (global mean 1.504, and replacing the
constant with per-group empirical means improves floor MSE by only 1.5% —
§2.6), so the uniform fold is both principled and empirically tight.

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
decisions are free. They are also *just-in-time*, not look-ahead: layer
L's hidden state exists only after layer L−1 completes, so without
cross-layer prediction there is no window in which to fetch a missing
expert. That is why a miss must be *served*, not awaited. A per-layer pool holds K experts' P1 (resident-P0 layout)
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

The governor is one instance of a more general property: the artifact is a
single file, and the *operating point* is chosen at runtime. The same 80B
artifact serves a knowledge profile (C=120: 10.2 GB, 28.1 tok/s, MMLU 86.0
and LAMBADA 78.2 intact, long-form reasoning broken), a reasoning profile
(C=240/K=32: 17 GB, 17–21 tok/s, MATH 64–65 / MBPP 81) and a prose gear
(C=290 all-P0: decode KLD −25%), and its refresh cadence trades prose
fidelity for speed continuously (KLD 0.774 → 0.303 at 17 → 9 tok/s, §2.7).
A static quantization fixes one of these points at build time; an
SSD-streaming engine fixes one per model. Because residency decisions are
just-in-time and free (§1.3), moving between points costs a refresh, not a
rebuild.

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

The 80B serves at 20–21.5 tok/s / 17.6 GB. The memory reduction itself
(42 → 17.6 GB) is MoE sparsity and decode locality at work — 47% of the
experts resident, the rest paged — and is not the contribution; what this
work adds is what happens on a miss (a floor, never a stall), the
stock-kernel planes, and the published accounting. Exact-mode evaluation — every slot
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

### 2.7 The decode regime: hostile metrics, the static-quant duel, and the levers that did not move

Exact prefill makes the teacher-forced KLD zero by construction (0.000 over
2,278 tokens), so the honest question moves to decode: what does the pool
cost when the served model must feed on its own tokens? We measure it
token-by-token with a live KV cache (exact prefix of 16 tokens, then one
token at a time, refresh on the production cadence), on WikiText 4×512 —
the dispersed domain where recency-based residency is weakest.

**Coverage is the first-order term.** On the 80B, per-token KLD against the
exact base is 1.563 at C=120 (23% of experts resident), 0.774 at the
production C=240 (47%), and 0.582 with the whole pool spent on P0 (C=290,
57%, no detail plane). That an all-P0 pool diverges *less* than the mixed
production pool is not a paradox: a miss outside the pool drops the expert
entirely, and a dropped expert costs far more than a 2-bit one, so ten
points of coverage outweigh the detail plane on dispersed prose. On focal
tasks the ordering reverses — the 2-bit gear loses 13 MATH points (iv
below). The 30B-Stream, whose P0 covers every expert (the
universal floor), measures 0.131 at comparable detail coverage — six times
lower than the 80B with drops. The per-position curves *decrease* with
length (80B C=240: 0.88 → 0.68 over 0–64 → 256–448 tokens; 30B-Stream
0.23 → 0.11): the damage is a start-up cost while the pool learns the topic,
not an accumulation in the KV cache. This retires the "poisoned cache"
hypothesis for both models.

**Cadence is the second-order term, and it is cheap only in KLD.** Refreshing
every 128/64/32 tokens instead of 256 brings the 80B's KLD to 0.566 / 0.416 /
0.303 (steady state 0.18), at 15.4 / 12.3 / 9.1 tok/s. Yet the focal-task
battery does not follow: with in-generation refresh every 128 tokens MATH-500
moves 64→65% and MBPP stays at 81% (n=100). On focal prompts the pool was
already tracking the routing; the KLD gain is spent on tokens the tasks do
not score.

**The duel.** Against Unsloth's UD-Q2_K_XL of the same Qwen3-Next-80B
[unslothqwen3next80b2026; unslothdynamic2026] (imatrix-calibrated,
per-layer dynamic ~3 bpw, 30.1 GB GGUF) on the same
machine, same prompts and parsers: the static artifact only runs with all
weights on CPU (`-ngl 0`, paging from disk; Metal OOM otherwise) at
12.6 ± 3.7 tok/s, and scores MATH-500 69% / MBPP 86% / MMLU 86.2% (n=100/
100/500). Stream at C=240 serves from 17 GB at 17.3–21.5 tok/s with an
exact prompt and scores 64–65 / 81 / 85.2. Our prior prediction that a
calibrated 2-bit would "lose 6–10 points" is falsified: task quality is at
par within n=100 noise on knowledge and nominally 5 points behind on the two
generative tasks. Stream's advantages are *system* advantages — fitting
where the static does not, +40–70% throughput, exact prefill, a floor and a
governor — not a quality lead over a well-calibrated static quantization
that needs 30 GB.

**What did not close the gap (all measured, all retired).** (i) *Absorb*:
re-injecting dropped gate mass into the shared expert, KLD 7.17 — the shared
expert's output scale is not that of a routed expert. (ii) *Overflow tier*:
incoming experts parked in 32 small rows with fast refreshes between full
ones, KLD 0.517 at 5.8 tok/s — the cadence benefit comes from refreshing the
large pool's membership and detail, and the extra gather per projection is
not free. (iii) *Thin universal floor* (25% width, 6 GB): 1.354 at C=120,
better than drops by 13% but it does not fit beside the production pool; a
floor below ~50% width sits "under the cliff", as on the 235B. (iv) *2-bit
gear in tasks*: C=290 all-P0 cuts KLD by 25% on prose but scores MATH-500
52% and MBPP 70% — spending the hot experts' detail on coverage costs
10–13 points on reasoning. The 2-bit configuration is a gear for general
text, not for focal generation.

**The rival's KLD on the same reference.** Feeding UD-Q2_K_XL the same
WikiText tokens through llama-server and scoring it against our stored exact
base (top-100-truncated KL, the standard approximation), the static 2-bit
measures **0.195** mean / 0.91 p95 / 1.94 p99 (0.189 on the positions our
curves use), flat along the sequence (0.27 → 0.18). Against our 0.774 at
production cadence and 0.303 at the cadence that halves our speed, the
calibrated static artifact is clearly more faithful on general prose; the
tails tell why (p99 1.9 versus 9.0): a drop outside the pool produces a
catastrophic token, a calibrated 2-bit never does. Only the universal floor
(30B-Stream, 0.131) goes below it — on a different model. The honest column
is therefore: at 24 GB the 80B is competitive on tasks and better as a
system, and *not* competitive on general-text fidelity.

Taken together, the ~5-point generative toll of the 80B at 24 GB is
explained neither by cadence nor by coverage-at-2-bits. What remains is the
quality of the 2-bit plane served on misses — P0 is a uniform-anchor plane,
the weakest possible floor — which points to a salience-protected master
(AWQ-style) as the only untested lever; we stopped here rather than keep
modifying the runtime without a clear return.

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
[llmflash2024]. The strongest Apple-native baseline is flash-moe
[flashmoe2026] — Qwen3.5-397B-A17B streamed from SSD on a 48 GB M3 Max at
4.4–5.5 tok/s with 2-bit experts and a custom C/Metal pipeline — and its
Anemll fork [anemllflashmoe2026] with GGUF-compatible 3-bit experts.
Neither has a miss-time floor (a miss waits for the SSD) nor published
quality accounting, and we have not yet dueled against them (§4).
Quality-elastic serving under pressure has a GPU-side analogue in
MorphServe [morphserve2025], which swaps less impactful dense layers for
quantized copies and resizes the KV cache under load; our governor resizes
*expert residency* — bit planes, not layers — on a single device. The
decode-routing locality our coverage laws rest on is independently
documented in the Qwen family: ELDR [eldr2026] exploits per-request expert
signatures for disaggregated serving, and ReMoE [remoe2026] fine-tunes
routers toward recently used experts (26% more reuse) — evidence that
stock routers are *less* local than residency would like, which is the
caveat our laws carry. HOBBIT [hobbit2024], MoE-Infinity [moeinfinity2024],
ExpertFlow [expertflow2024], ProMoE [promoe2024], Fiddler [fiddler2025],
KTransformers [ktransformers2025] and MoE-Lightning [moelightning2025]
offload whole experts on CUDA/x86. **The nested-quantization lineage** ("one model, many precisions") is
saturated and we claim none of it: Any-Precision LLM [anyprecision2024]
serves bit-planes through a custom CUDA engine over non-uniform codes;
MatQuant [matquant2025] co-trains MSB-sliceable weights (its right-shifted
distributions are the training-time answer to the truncation failure we
anatomize in §1.1); MatGPTQ [matgptq2026] brings that to one-shot PTQ with
dedicated kernels; AnyBCQ [anybcq2025] operates directly on binary-coded
planes; D²MoE [d2moe2025] nests expert bit-widths (MWQ) on-device — with a
bespoke dequantization kernel, which is exactly the contrast that motivates
us: our planes are *standard affine tensors*, and every level is served by
the unmodified stock kernel. AWQ [awq2024] states the salient-weights
principle our truncation anatomy reaches through the grid; SqueezeLLM
[squeezellm2024] establishes the memory-bound batch-1 regime this whole
design lives in, and MoQE [moqe2023] documents that expert FFNs tolerate
2-bit — the reason a 2-bit floor is viable at all when gate-protected.
Earlier offloading lines are mixtral-offloading [mixtraloffload2023],
AdapMoE [adapmoe2024] (gate-aware skipping) and DynaExq [dynaexq2025]
(hot-expert precision promotion). An independent corroboration of our sync
finding exists in llama.cpp's expert-cache RFC [llamacpp24528]: their Metal
slot-pool experiments ran 2× slower than vanilla even at 97–99% hit rate
purely from per-layer sync points — the same wall our dynamic-biases
membership removes — and their hybrid hit/miss execution is exact-only (no
precision floor). The kimi-k3 MLX port [kimik3mlx] stores two quantized
banks per expert and reportedly pays a host sync to split indices — the
design point our single-kernel path avoids.

**Static salience-guided quantization as the deployment rival.** Unsloth's
Dynamic GGUFs — the 80B artifact we duel is of Dynamic 2.0 lineage per its
model card [unslothqwen3next80b2026]; the Dynamic 3.0 write-up
[unslothdynamic2026] is the methodology reference — allocate bits per
layer/tensor from
imatrix calibration (no QAT), producing a family of artifacts whose
quality/size point is chosen at download time and whose loss is permanent
and uniform in time. For the 80B-on-24GB user the honest alternative is
therefore not "it doesn't fit" but a ~2-bit UD artifact that fits and runs
natively in llama.cpp; their own Gemma-3 27B table prices that route
(MMLU 68.70 at Q2_K_XL vs 71.47 at Q4_K_XL — a well-made static 2-bit pays
~3 points everywhere, always). Our thesis is that temporal bit allocation —
the hot path always at full 4-bit, only the cold tail paying the floor —
dominates static downsizing at equal memory; the side-by-side UD-Q2 baseline
on the same machine is on our pre-submission roadmap, and we note the
symmetry honestly: an imatrix-informed static 2-bit is a far better 2-bit
than our naked P0 — our defense is that P0 is almost never served, not that
it is a good 2-bit. From the same line we adopt the metrics critique
[accuracynotall2024]: perplexity averages can cancel token-level damage,
and KL-divergence plus answer flips are the better-correlated measures —
directly relevant to us, since our strongest statistical evidence is a PPL
equality. Measuring served-vs-base KLD and greedy trajectory divergence
(we hold the exact-mode base control that makes this cheap) is queued as
a headline metric alongside PPL.

**Phase-asymmetric fidelity.** Prefill/decode disaggregation is standard for
throughput [distserve2024]; PMPD [pmpd2024] lowers precision progressively
along decode, and HMA-Serve [hmaserve2026] serves prefill in vendor-native
low precision with BF16 decode. Our exact prefill points the asymmetry the
other way — exact prompt, pool-governed decode — motivated by a measured
toll (+28%/+120% PPL from flat prefill routing, §1.4) rather than a
throughput budget.

The static companion is Topiary [luque2026topiary]. Speculative decoding on hybrid SSM architectures is an
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

**The duel's baseline.** CPU-only is the only way the 30 GB GGUF runs on
24 GB, so its 12.6 tok/s is near that file's realistic ceiling here rather
than a straw man — but the strongest Apple-native rival for this exact
problem, flash-moe/Anemll (SSD-streamed experts), has not been dueled. A same-model duel is
not currently possible: both engines are single-architecture (Qwen3.5-397B-
A17B only: 60 layers, 45 GatedDeltaNet + 15 attention, 512 experts), with
no path for a Qwen3-Next-80B or a 35B; a cross-model comparison (their 397B
streamed from SSD on our 24 GB machine) would measure platforms, not
methods. What can be stated from their published numbers: 4.4–5.5 tok/s
(original, 48 GB M3 Max, 2-bit experts) to 12.9 tok/s (Anemll fork, Q3
experts, 128 GB M5 Max), expert I/O blocking on every miss (47% of per-token
decode time), and quality reported qualitatively ("breaks JSON/tool calling"
at 2-bit) — no perplexity or KLD. The platform argument therefore rests on
those three structural differences, not on a head-to-head.

**Statistical power.** A 15/15 at n=15 bounds the failure rate only below
~20% (rule of three): it is a "no detected degradation" statement, not
evidence of parity. Task accuracies at n=15/25/50 carry wide intervals
(95% CI ≈ ±11 points at 92%/n=25, ≈ ±7.5 at n=50): they support
*indistinguishability* claims and large effects (the −20-point coverage
break), not fine rankings. Perplexity equalities (base matched to four
decimals) are the statistically strong results — means over tens of
thousands of tokens. Scaling tasks to n≥300 with bootstrap CIs is the
pre-submission bar we set ourselves.

**Generality.** Single machine, single routed family (Qwen) plus OLMoE as a
bench. Qwen3's decode routing is highly local; a load-balanced router
(OLMoE, Mixtral) could defeat recency-based residency and must be measured
before the coverage laws are claimed beyond "MoE with high decode locality".

**Engineering surface.** The fast-path depends on one mlx-lm internal
(`switch_layers._gather_sort`) and on `gather_qmm`'s `sorted_indices`
semantics; CI is pinned to the tested versions (mlx 0.32.0, mlx-lm 0.31.3)
and the package declares bounded compatible ranges; the helper is
small enough to vendor if 3.x breaks it. Cold-boot throughput is now
measured (fresh reboot, interleaved rounds): the 35B serves 47.2 tok/s
median from cold — round 1 within 1% of warm — so pool warm-up is
negligible at this scale; the 80B's cold rounds triggered swap (1.1 GB)
and are discarded under our swap-zero protocol, leaving its warm figure
(17.3–21.5 tok/s) as the citable one pending a settled-system re-run.

Also: PPL anchors for
the 35B under a short protocol; the 235B artifact mixes DWQ with plain 4-bit
siblings. Exact prefill removes
the pool's teacher-forced toll by construction; the decode-side toll is now
quantified (§2.7) but only on WikiText 4×512 and a 100-item generative
battery — the duel's 5-point gap is within n=100 noise on each task taken
alone, and only the direction (consistent across MATH-500 and MBPP) is
claimed. In the solutions bench, only the
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
- **Servable artifacts** (Hugging Face, public):
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
