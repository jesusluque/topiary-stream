# 7. Catalogue of measured negatives

Each entry: the idea, the measurement that killed it, and the lesson. The
negatives are documented with the same rigor as the positives because they
bound the design space just as well.

## 7.1 Truncation-derived pyramid

- **Idea**: derive Q4/Q2 from a Q8 master by bit shifting (a single copy,
  nested levels for free).
- **Measurement**: wins on weight L2 and max error; derived Q2 ~20× worse PPL
  than a native Q2 (OLMoE-7B).
- **Lesson**: the truncated grid's error lands on the salient weights; the
  pyramid is anchored at the serving level and refined upward.

## 7.2 Empirical per-group centroid (`--centroid empirical`)

- **Idea**: replace the uniform 1.5·s fold with the real per-group mean of
  the discarded plane (same format, only the constant changes).
- **Measurement**: floor MSE improves by 1.5%; global mean 1.504.
- **Lesson**: the low 2 bits are near-uniform noise; there is no information
  to recover there. The floor improves through salience (which weights), not
  through constants (which values).

## 7.3 Delta compression between rows

- **Idea** (external): a 1-bit mask of changes between neighboring rows +
  the changed values, like a video codec.
- **Measurement**: equality between neighboring rows or neighboring experts =
  exact chance rate (8.6% measured vs 8.6% from the q4 code distribution);
  the scheme expands to 4.66 bits/weight.
- **Lesson**: trained quantized weights have no row-to-row redundancy; the
  exploitable structure is *which* weights matter.

## 7.4 Salient P1 subsampling as a general optimization

- **Idea**: serve P1 only for the salient prefix (4:2:2, 2:2:4…).
- **Measurement**: MATH nearly flat from 3.0× to 1.0× bytes (67→64), but the
  full battery for 2:2:4 pays −8 MBPP and −3.6 MMLU.
- **Lesson**: an emergency memory dial (0:0:4 serves usable MATH at 10.4
  GB), not a default. Validate per model: in non-Topiary checkpoints the
  channels do not come ordered.

## 7.5 Pool-served prefill

- **Measurement**: +6–11% PPL on the 35B, +28%/+120% on the 80B; 80B tasks at
  84%/96%.
- **Lesson**: prefill routing is flat and defeats recency; serving the prompt
  exact (it is one batched pass) erases the whole toll. This is the negative
  that became a design.

## 7.6 Thin universal floor

- **Measurement**: 80B at 25% width (6 GB, captures 61.6%): KLD 1.354 at
  C=120 (13% better than drops, but below the cliff); does not fit alongside
  the C=240 pool (thrashing, 0.1 tok/s). 235B at 16.7% (53.5%): degenerate
  text.
- **Lesson**: the floor's width must capture ≥85–90% of the salience (≈50%
  width in these models) to be a quality level; below that, last-resort
  coherence.

## 7.7 Token-level retry-on-miss (235B)

- **Idea**: retry the token when an expert is missing, hoping it converges
  with C ≫ working set.
- **Measurement**: 99% retries.
- **Lesson**: the condition is `L·k·P(miss) ≪ 1`, unreachable in deep models
  with load-balanced expert tails.

## 7.8 Absorb (the shared expert as floor)

- **Idea**: transfer the gate mass of dropped experts to the shared expert,
  already resident and trained.
- **Measurement**: KLD 7.17 (flat ~7 along the sequence).
- **Lesson**: the output scales of the shared expert and of a routed expert
  are not comparable; it over-weights the shared expert and destroys.

## 7.9 Overflow tier (cheap refresh, `--ovf-merge`)

- **Idea**: incoming experts go to 32 small rows (~27 MB copies) at a fine
  cadence; the big pool is only refreshed every N fast ones.
- **Measurement**: KLD 0.517 at cadence 32 (plain refresh 32: 0.303; 128:
  0.566) and a clean 5.8 tok/s (plain 8.5 at 32; 17.1 at 256).
- **Lesson**: the benefit of cadence comes from refreshing the **big** pool
  (membership and P1 detail); the tier adds one `gather_qmm` per projection
  and token, and 7 fast refreshes per full one do not come for free.

## 7.10 2-bit mode (C=290, K=1) for tasks

- **Idea** (dynamic LOD): spend the detail bytes on coverage.
- **Measurement**: prose KLD −25% (0.774→0.582), but MATH 52 (vs 65) and MBPP
  70 (vs 81).
- **Lesson**: in the focal regime the 4-bit detail of the hot experts is
  worth more than +50 experts at 2 bits. A gear for general text, not for
  reasoning.

## 7.11 Cadence as a task remedy

- **Measurement**: refresh 256→32 lowers the KLD 0.774→0.303; refresh 128
  within generation leaves MATH 65 / MBPP 81 (no change); cost −10/−28/−47%
  tok/s.
- **Lesson**: on focal prompts the pool was already tracking the routing;
  the KLD improves on tokens the tasks do not score. Cadence is a lever for
  prose fidelity, with a price.

## 7.12 The poisoned-cache hypothesis (decode)

- **Measurement**: per-position curves decrease (80B 0.88→0.68; 30B
  0.23→0.11).
- **Lesson**: the damage is start-up damage (the pool does not yet know the
  topic); it does not accumulate in the KV. Retired.

## 7.13 The prediction "static 2-bit loses 6–10 points"

- **Measurement**: Unsloth UD-Q2_K_XL 69/86/86.2 vs our 65/81/85.2; KLD 0.195
  vs 0.774.
- **Lesson**: a per-layer dynamic 2-bit with imatrix (~3 bpw, sensitive
  layers protected) holds quality. Stream's advantage is a systems one
  (§1.3).

## 7.14 What never got measured (and is not claimed)

Burst after the prompt, two-gear governor, EMA by mass, warm-up with
runner-up experts, margin sensor, per-layer selective refresh, BF16 router /
8-bit skeleton. Implemented; cancelled under the rule "no clear return, or
drops on other axes, means we do not continue".

## 7.15 The residue

After ruling out cadence and coverage, what explains the ~5-point generative
toll and the KLD tails (p99 9.0 vs 1.9) is the quality of the 2-bit plane
itself served on misses: P0 is a uniform anchor plane, the worst possible
floor. The one untried lever is a salience-protected master (affine
AWQ/imatrix) — parked for disk-space reasons and by the user's decision.
