# 1. Vision and lineage

## 1.1 What problem it solves

A 4-bit quantized MoE that does not fit in a laptop's unified memory cannot
be served: the loader brings it in whole or not at all. On a 24 GB MacBook
Pro M5 Pro that rules out Qwen3.5-35B (19.5 GB: it loads, but Metal dies on
generation), Qwen3-Next-80B (42 GB) and Qwen3-235B (132 GB).

Topiary Stream is a **residency runtime**: the full model lives on disk in a
per-expert pageable format, and RAM holds only (a) the non-expert skeleton
and (b) a per-layer *pool* with the experts the router is using, at the
precision the router justifies. The result, measured on that machine:

| Model | 4-bit size | Served peak | tok/s | Quality |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | 19.5 GB | 12.5–14 GB | 44.5–47 | PPL = base; HumanEval/GSM8K 92/92 (n=25/50) |
| Qwen3-Next-80B-A3B | 42 GB | 17–17.6 GB | 17–21.5 | PPL = base; MMLU 85.2 (n=500); MATH-500 64–65, MBPP 81 (n=100) |
| Qwen3-235B-A22B | 132 GB | — | — | no servable middle at 24 GB (mapped negative) |

All of it with **stock MLX quantized kernels** (`gather_qmm`): no custom
Metal, no dedicated inference engine.

## 1.2 Lineage

The project has three layers, each with its own repository:

1. **nanite-moe** (the lab, private). It was born as "hierarchical expert
   LOD by gate", inspired by Nanite: camera = hidden state, distance =
   router score, LOD = the precision at which the expert is read. Phases
   F0–F48 with numbered reports in `reports/` (architectures, routing
   locality, dual-precision quality, rank axis, width, nesting, clusters,
   composition, scale, runtime, final suite, prior art, audit, simulated
   streaming, external confrontations). That is where the laws Stream
   exploits were measured: routing locality in decode, exact decomposition
   of a 4-bit tensor into two 2-bit planes, the correct direction for
   building the pyramid, token-to-token persistence of the routing (f48).
2. **Topiary** (public: code and `qwen3-30b-topiary*` checkpoints on HF).
   Static sculpting by salience: expert-width pruning with channels
   reordered by salience (the "taper"). It produces smaller and somewhat
   faster checkpoints; its price, measured afterwards, is knowledge
   (−10 MMLU), not reasoning.
3. **Topiary Stream** (this repository; private as of today). The residency
   runtime. It inherits from Topiary the salience ordering of channels
   (which makes the "salient prefix" of the P1 plane contiguous) and from
   the lab the anchored pyramid and the router telemetry.

## 1.3 The thesis, and its correction

Initial thesis: *gate-guided temporal bit allocation dominates static
downsizing at equal memory*. Measured against the best static option
available (Unsloth UD-Q2_K_XL of the same 80B, imatrix, ~3 bpw, 30 GB), the
thesis stands **corrected**:

- On tasks, the calibrated static is at par or nominally ahead (MATH 69 vs
  65, MBPP 86 vs 81, MMLU 86.2 vs 85.2; n=100/100/500).
- On general-text fidelity (KLD against the exact base) the static wins
  clearly (0.195 vs 0.774 in production).
- What Stream contributes is **at the system level**: it fits where the static does not
  (17 vs 30 GB), it is 40–70% faster (GPU versus CPU paging), it serves the
  prompt bit-exact, it has a guaranteed floor and an elastic governor.

The honest formulation: *temporal bit allocation is the way to run a 42 GB
model well in 24 GB; it is not a way to beat, on quality, a well-calibrated
static that needs 30.*

## 1.4 The three ideas, one sentence each

1. **Anchored bit-plane pyramid.** A 4-bit affine tensor is exactly two
   2-bit tensors (`q4 = 4·q_hi + q_lo`), both valid for the stock kernel;
   P0 (high bits, with a centroid bias) is a servable floor at half the
   bytes; P0+P1 is the bit-exact 4-bit.
2. **The gate governs residency.** The router scores exist before a single
   expert byte is read; the residency decision is free. Pool membership is
   encoded in a dynamic *biases* tensor, so a single kernel call serves hot
   and cold without a CPU/GPU sync per token.
3. **It breathes.** The governor reads macOS's real memory pressure at every
   refresh and resizes the pool; a miss never blocks (it serves the floor,
   "blurry for one frame") and the pool learns it at the next refresh.

## 1.5 Glossary

| Term | Meaning |
|---|---|
| **P0 / P1** | High / low bit planes of a 4-bit code. P0 alone = 2-bit level (floor); P0+P1 = exact 4-bit. |
| **Anchored pyramid** | Native Q4 as anchor; Q8 = anchor + refinement plane; Q2 = truncated anchor (only for cold slots protected by the gate). |
| **Centroid fold (1.5·s)** | Constant added to the bias when P0 alone is served: the expectation of the discarded plane under uniform q_lo. Measured nearly uniform (mean 1.504). |
| **resident-p0** | Artifact layout in which P0 lives in the checkpoint and P1 in memmaps per (layer, projection). For models whose P0 fits in RAM (35B, 30B). |
| **full-memmap** | Layout in which P0, P1 and scales/biases all go to memmaps per expert row; the checkpoint is a 3–5 GB skeleton. For 80B and 235B. |
| **Pool** | Per-layer resident set: K experts with P1 (resident-p0) or C experts at P0 of which K with P1 (full-memmap). |
| **C / K** | Pool size at P0 / number of experts with P1 detail. 80B production: C=240, K=32 (out of 512 experts/layer). |
| **Refresh** | The only deferred synchronization: drains the routing counters into an EMA, recomputes membership and pages in the incoming rows. Default cadence 256 tokens. |
| **Miss / drop** | A routed expert that is not in the pool. In `nosync` it is removed and the remaining gates are renormalized (drop); in `floor*` its floor is served. |
| **Universal floor (floor2d)** | P0 × salient prefix of all experts, resident, so that no slot ever drops. Viable if salience is concentrated (≥85–90% captured). |
| **Exact prefill** | Any forward with T>1 is served at full 4-bit (P1 of the expert union read once from the memmap). The pool policy governs decode only. |
| **Governor** | Pressure→K loop: reads `vm_stat` at every refresh and shrinks/grows K between bounds. |
| **Taper** | Topiary's salience-based width pruning (reordered channels). |
| **Orders / routed salience** | `E[h²]·‖W_down[:,i]‖²` per (layer, expert, neuron), computed through the pager without a checkpoint; pool prior and floor input. |
| **Decode KLD** | KL(base ‖ served) token by token with a live KV cache, exact 16-token prefix, refresh at production cadence. The hostile metric of the pool's toll. |
| **Coverage law** | Measured relation between the fraction of resident experts and damage: 23% breaks long reasoning; 47% serves; 100% at P0 (universal floor) reduces KLD 6×. |
