# 2. Method

## 2.1 Anchored bit-plane pyramid

### The exact identity

MLX quantizes in groups of `GROUP = 64` values with the affine form
`w = s·q + β` (scale `s` and bias `β` per group; `q ∈ [0, 15]` for 4 bits).
Writing the code as `q4 = 4·q_hi + q_lo` with `q_hi, q_lo ∈ [0, 3]`:

```
w = s·(4·q_hi + q_lo) + β = [4s]·q_hi + [s]·q_lo + β
```

That is, a 4-bit tensor is the **sum of two valid 2-bit tensors**:

| Plane | Codes | Scale | Bias |
|---|---|---|---|
| P0 | `q_hi = q4 >> 2` | `4·s` | `β` |
| P1 | `q_lo = q4 & 3` | `s` | `0` |

Each one is served with `mx.gather_qmm(..., bits=2, group_size=64)` exactly
as shipped in MLX. The P0+P1 reconstruction through the kernel is bit-exact
(test `test_plane_reconstruction_exact_through_kernel`: error 2.6e-7, that
of float rounding).

### The floor: P0 alone with centroid fold

Serving P0 without P1 leaves out `s·q_lo`. With `q_lo` uniform over
{0,1,2,3} its expectation is `1.5·s`, so the floor uses bias `β + 1.5·s`
(test `test_cold_centroid_bias`). It is the first-order unbiased constant;
we measured that the discarded plane is nearly uniform in practice (global
mean 1.504) and that replacing it with the empirical per-group mean
(`--centroid empirical`) improves the floor's MSE by only 1.5% — a
documented negative (§7).

### Direction of construction (what it took a reversal to learn)

A pyramid can be derived by **truncating downward** from a fine master (Q8 →
Q4 → Q2 by bit shifting) or by **anchoring** it at the serving level.
Truncation wins on every uniform weight metric (L2 and max error) and loses
end-to-end: on OLMoE-7B the derived Q2 was ~20× worse in PPL than a
natively fitted Q2. The truncated grid concentrates its error on the
salient weights, which uniform metrics under-weight.

Rule adopted (`pyramid.py --stage anchored`):

| Level | Construction | Quality |
|---|---|---|
| Q4 | native min/max fit from the 8-bit master | exact serving quality |
| Q8 | Q4 anchor + 4-bit refinement plane (`codes8 = q4·16 + q_lo4`; `>>4` recovers Q4 exactly) | PPL 3.9209 vs true master 3.9213: free |
| Q2 | truncated anchor (`q4 >> 2`, `s·4`, `β + 1.5·s`) | floor, **only** for cold slots protected by the gate |

`--stage derive` (naive truncation) and `--stage requant` (native refit to
N bits) are kept as builders for the negative control and the baseline.

### Scope of the format

The exact split requires **flat affine** quantization (`s·q + β` per group).
It does not port to GGUF's strong formats: K-quants nest 6-bit scales in
superblocks and IQ-quants are codebook-based. Only block-wise affine of the
Q4_1 kind is structurally compatible. It is the flip side of the "stock
kernels" property and is declared as such. (Upward it does port: Q8_0 →
2×Q4_0 is symmetric with no fold.)

## 2.2 Artifacts: two layouts

`split.py` converts a community 4-bit checkpoint (`mlx-community/…`) into a
self-contained directory (config, tokenizer, planes), which is what gets
uploaded, downloaded and served.

### `resident-p0` (P0 fits in RAM: 35B, 30B)

- The expert tensors (`*.switch_mlp.{gate,up,down}_proj.weight`) are
  replaced in the checkpoint by their P0 plane (2 bits); `config.json`
  gains per-module overrides `{"group_size": 64, "bits": 2}` and the marker
  `stream_layout = "resident-p0"`.
- P1 goes to memmaps `L{li}.{proj}.p1.bin` (`uint32`, `[experts, words]`)
  with `p1_manifest.json` (experts, out, cols, words, layer, proj).
- Scales and biases stay in the checkpoint (they are the original 4-bit
  ones; the runtime multiplies `s·4` for P0).
- RAM at load ≈ 60% of the 4-bit model (35B: 19.5 → ~12 GB).

### `full-memmap` (P0 does not fit: 80B, 235B)

- Nothing expert-shaped remains in the checkpoint: `L{li}.{proj}.p0.bin`,
  `.p1.bin` and `.sb.bin` (stacked scales and biases, `float16`,
  `[experts, 2, sb_cols]`), pageable per expert row.
- A **skeleton** (`model.safetensors`, 3–5 GB) with the non-expert weights
  and *dummy* 1×64 switch tensors; `config.json` carries `stream_skeleton =
  true`. At load, `pager.maybe_patch_skeleton` patches the MoE block
  constructor (`Qwen3MoeSparseMoeBlock`, `Qwen3NextSparseMoeBlock`) so that
  it instantiates `SwitchGLU(hidden, 64, 1)` and the checkpoint matches.
- `stream_manifest.json` describes each `L{li}.{proj}`: experts, out, cols,
  words, sb_cols, s_shape.

`--consume` deletes each source shard (the symlink and the blob in the HF
cache) after processing it, capping peak disk at ~checkpoint + one shard of
planes. The skeleton is written **first**, so the runtime never needs the
consumed source — a lesson paid for with a 132 GB re-download.

## 2.3 Gate-governed pool, without synchronization

### The structural advantage

In an MoE block the output is `y = shared(x) + Σ_{i∈topk} g_i·E_i(x)`. The
router (`gate`) is a tiny matrix that is evaluated **before** the experts
are read: the "camera distance" is known a priori, for free. Stream uses
that information to decide which experts are resident and at what
precision, without predicting anything.

### Membership policy, not a per-token threshold

Per layer, the pool holds:

- resident-p0: K experts with their P1 plane on GPU (`pool [K, out, in_words]`,
  `pool_s [K, …]`); P0 of all experts is resident in the checkpoint.
- full-memmap: C experts at P0 (`p0 [C, …]`, `s0`, `b_dyn`) and K of them
  with P1 (`p1 [K, …]`, `s1`).

Membership is encoded in a **dynamic biases tensor** `b_dyn`: members with
P1 carry `β` (their P1 completes them to exact 4-bit); the rest carry
`β + 1.5·s` (floor). Thus **one** `gather_qmm` call over P0 serves hot and
cold, and a second one over the P1 pool (with remapped indices and a
membership mask) adds the detail to those that have it. There is no
per-token `if` and no CPU↔GPU copy per layer.

This was the difference between 8.1 and 44.5 tok/s on the 35B: the
reference version synchronized ~40 times per token (once per layer to query
membership); the fast path synchronizes not once.

### Refresh: the only deferred synchronization

In the forward, the top-k indices are stacked into `pending[layer]` (lazy,
unevaluated in decode). Every `REFRESH` tokens (256 by default; 128 in the
batteries with `--gen-refresh 128`), `refresh_all()`:

1. Concatenates and evaluates the pending indices, sanitizes them
   (`0 ≤ idx < E`: hardening against garbage indices under memory pressure).
2. Updates the per-layer EMA: `ema = α·ema + counts` (α = 0.7 resident-p0,
   0.8 full-memmap) — frequency + recency, the signal that predicts MoE
   locality.
3. Recomputes the top-C / top-K and pages in **only the incoming rows** from
   the memmap (`np.memmap` reads only the pages touched).

In resident-p0 the pool is re-paged whole on any change (K=32 full rows are
~40 MB from page cache; refresh=64 cost only −2.5% tok/s). In full-memmap
the rows are ~10× larger and a full rebuild cost 10× in decode (~13 GB of
copies every 64 tokens): the refresh is **incremental**, with bounded churn
(`MAX_CHURN = 8` incoming per refresh, displacing the 8 with the lowest EMA)
and only the diff of the P1 rotation is touched. `mx` `setitem` copies the
whole buffer, hence the obsession with touching little (~1.7 s per full
refresh on the 80B).

### A miss never blocks

If a routed expert is not in the pool:

- `nosync` (production): it is removed and the remaining gates are
  renormalized (*drop-renormalize*). Fast; the damage grows with the
  domain's dispersion.
- `floor`: its P0 is fetched from the memmap synchronously. Correct but tied
  to the sync: ~0.2 tok/s with 94 layers. Demonstration mode.
- `floor2d`: its slice of the resident universal floor is served (§2.5).
- `absorb`: the shared expert absorbs the dropped mass. **Resounding
  negative** (KLD 7.17): the output scales are not comparable.
- `exact`: every slot is served P0+P1 straight from the memmap (== true
  4-bit); batch/teacher-forced only (~4 GB peak). It is how the true base
  of a model that does not fit is measured.

## 2.4 Exact prefill

Any forward with `T > 1` (the prompt, or a teacher-forced chunk) is a
batched pass: P0 is read as always and the P1 of the **union** of experts
the prompt needs is fetched once from the memmap
(`np.unique(inds)` → `rme` remapped indices → two `gather_qmm`). The pool
policy therefore governs decode only.

It is the design decision that erased the runtime's largest measured toll:
pool-served prefill cost +6–11% PPL on the 35B and +28%/+120% on the 80B
(prefill routing is flat and defeats recency-based membership), and left
decode a poisoned KV cache. With exact prefill: served PPL = base to four
decimals, teacher-forced KLD = 0.000, and the 80B went from 84%/96% to
15/15 / 15/15 on HumanEval/GSM8K.

`serve.py` additionally does a **double prefill**: one pass of the prompt
only for routing statistics (the pool adapts), and the real pass against an
informed pool. Without it a cold uniform pool served the prompt degraded
(measured on the 235B).

## 2.5 Universal 2D floor (`floor.py` + `floor2d`)

Each expert contributes its `k_floor` most salient neurons (by routed
*orders*) at the P0 level, packed into a resident `safetensors`. With the
floor loaded no slot ever drops: experts outside the pool serve their slice.
For `down_proj` the cut is by groups of 64 input columns (consistent with
group-wise quantization); for `gate/up` by output rows.

Honest measured guidance: the floor's quality follows the **concentration
of salience**. With the prefix capturing >85–90% of the energy it is a real
quality level (30B-Stream with P0 of all experts: KLD 0.131); at ~50%
(235B: 53.5%; 80B at 25% width: 61.6%) it degenerates to last-resort
coherence and may not be servable (235B: degenerate text; 80B: 1.354 at
C=120 and it does not fit alongside the C=240 pool → thrashing).

## 2.6 Elastic governor

Residency is a dial, so it carries a controller. At every refresh,
`available_gb()` reads `vm_stat` (free + inactive + purgeable +
speculative × page size) and `govern(low=4, high=7, k_min=4,
k_max=48, step=8)` shrinks K if `avail < low` or grows it if `avail > high`;
the next refresh materializes the change (the pools are rebuilt at
`pool_k` by design). Measured under an 8 GB external balloon
(`examples/balloon.py`): four automatic shrink steps (K 32→4, active memory
13.7→12.8 GB) with generation completing cleanly; and with the governor
active the 35B delivered 51.1 tok/s (it probably shrank K on its own).

## 2.7 Implemented variants

**Measured (and with a verdict):**

- `--p1-frac g,u,d`: serve P1 only for the salient prefix of each
  projection (a contiguous slice because Topiary checkpoints ship their
  channels ordered by salience; `0` = projection at pure floor). Verdict:
  memory emergency dial, not a general optimization (§6.3).
- `--centroid empirical`: negative (1.5% MSE).
- Refresh cadence (`--refresh`, `--gen-refresh`, `--kld-refresh`): THE KLD
  lever, does not move tasks; cost in tok/s (§6.5).
- Overflow tier (`--ovf-merge N`): negative (§7).
- 2-bit mode / second gear (high C, K≈1): negative on tasks, −25% KLD on
  prose (§6.5, §7).

**Implemented as flags but NOT measured** (the campaign was cancelled
before; nothing is claimed for them): refresh burst after the prompt
(`--burst-len/--burst-every`), two-gear governor (`--gear`,
`--gear-hi/lo`, `miss`/`margin` sensor), gate-mass-weighted EMA
(`--ema-mass`), overflow prewarming with runners-up
(`--prewarm N`), per-layer selective refresh (`--refresh-min-miss`),
tensor protection (`protect.py`: router to BF16, skeleton to 8 bits).
They remain in the code because they are small and isolated; their default
value disables them.
