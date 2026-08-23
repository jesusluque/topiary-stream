# 3. Runtime internals

A walk through `src/` in execution order. None of this depends on the
model's specific module names: MoE blocks are discovered by content.

## 3.1 `common.py` — shared helpers

- `set_seeds(1234)`: `mx.random.seed` + `np.random.seed`. With greedy
  decoding the seeds only affect the benchmarks' sample selection.
- `load_corpus(path, limit_tokens)`: reads `jsonl` with `{"text", "n_tokens"}`
  up to a token budget.
- `token_nll(logits, ids)`: teacher-forced per-token NLL with direct
  log-softmax `z − logsumexp(z)` (stable with 248k vocabularies; `log(softmax
  + eps)` lost the tails).
- **MoE block discovery** (`find_moe_blocks`): walks `model.layers`
  (unwrapping `language_model` if present) and accepts as a block any module
  — or the layer itself — that has a router child (`gate|router|gate_proj|wg`)
  and an expert container (`switch_mlp|experts|mlp|moe`) with at least one
  3-D tensor `[n_experts, out, in]`. Returns `[(layer_index, block)]`.
- **Plane arithmetic**: `unpack4` (packed uint32 → codes 0..15; MLX packs
  little-endian, low nibble first), `pack2` (2-bit codes → uint32, 16 per
  word), generic `pack`/`unpack_bits` for 2/4/8 bits (tests
  `test_unpack_pack_roundtrip`, `test_unpack_bits_order`).

## 3.2 `fastpath.py` — the `resident-p0` runtime

### Per-layer state: `FastLayer`

Built in `patch_fast` for each MoE block with its entry from
`p1_manifest.json`:

| Field | Content |
|---|---|
| `projs[name]` | per projection: `p0` (the checkpoint `weight`, already 2-bit), `s`, `b`, `b_cold = b + 1.5·s`, `mm` (P1 memmap), `out`, `b_dyn`, `r` (P1 rows/words served), `pool`, `pool_s` |
| `pool_k` | current K (the governor changes it live) |
| `lookup` | `int32[E]`: position in the pool or −1 |
| `members` | list of experts with P1 |
| `ema` | `float64[E]` |

With `--p1-frac`, `r` trims: gate/up by output rows (multiple of 64); down
by groups of input words (`rg·4` words = `rg` groups of 64). `frac = 0`
leaves `r = 0` and the projection lives on the floor alone.

### Patched forward

`patch_fast` replaces `block_cls.__call__`:

1. `gates = softmax(gate(x))` in float32 (`precise=True`); `inds =
   argpartition` top-k (unsorted — the same set as the original); `scores`
   normalized if `norm_topk_prob`.
2. In prefill (`x_flat.shape[0] > 1`) `inds` is **evaluated right away**
   (lazy indices under memory pressure produced garbage) and served exact
   (§2.4): `pfx` = `gather_qmm(P0, s·4, b)` + `gather_qmm(P1_union, s_u, 0)`.
3. In decode: `inds` is stacked into `STATE["pending"]` **without
   evaluating**; if `inds.size ≥ 64`, `switch_layers._gather_sort` is used
   (sorts tokens by expert for the kernel with `sorted_indices=True`) and
   `_scatter_unsort` on the output. Per projection: `y = gather_qmm(P0, s·4,
   b_dyn, idx)` + `gather_qmm(pool, pool_s, 0, remap)·mask`, where `remap =
   max(lookup[idx], 0)` and `mask = lookup[idx] ≥ 0`. With `r < out` on
   gate/up the result is padded with `mx.pad`; with down the input is
   trimmed `xin[..., :r·16]`.
4. The original block's GLU activation, weighted sum by `scores`, plus the
   shared expert if present (`sigmoid(shared_expert_gate(x)) ·
   shared_expert(x)`).

### Refresh and governor

- `refresh_all()`: per layer, concatenates the pending indices, sanitizes
  them and calls `FastLayer.refresh(counts)`: `ema = 0.7·ema + counts`,
  `want = top-K`; if nobody new comes in, nothing to do; if someone does,
  re-pages the whole pool from the memmap (small rows), recomputes `pool_s`,
  `lookup` and `b_dyn = b_cold` with `b` on the members. Returns the number
  of newcomers.
- `available_gb()` and `govern(...)`: see §2.6. Returns a message
  `[gov] avail X GB -> K a->b` when it changes and clears `members` to force
  a rebuild at the next refresh.

## 3.3 `pager.py` — the `full-memmap` runtime

### Loading

`load_model(artifact)`: if `config.stream_skeleton`, patches the
constructors (`maybe_patch_skeleton`) and loads the whole skeleton;
otherwise, loads lazily and evaluates everything except the switch tensors
(the path for models without a skeleton).

### Per-layer state: `PoolLayer`

| Field | Content |
|---|---|
| `mm[proj]` | memmaps `p0`, `p1` (`uint32 [E, words]`), `sb` (`float16 [E, 2, sb_cols]`), `out`, `s_shape` |
| `c`, `k` | pool sizes |
| `ema` | initialized with the **prior** (`orders_routed.npz`: `salience_li.sum(axis=1)`) or uniform |
| `members0/1`, `lookup0/1` | P0 / P1 membership and positions |
| `pools[proj]` | `p0 [C,…]`, `s0`, `b_dyn`, `p1 [K,…]`, `s1`; plus the overflow tier `po/so/bo [OVF=32,…]` |

`_install(want0)` does the full rebuild (only at init and on a gear
change); `refresh(counts)` is the incremental one from §2.3. `_rows(proj,
plane, ids)` and `_sb(proj, ids)` read rows from the memmap and convert
them to `mx.array` with shape `[n, out, -1]`.

### Patched forward, by mode

1. Routing as in fastpath. With `--prewarm N` or the `margin` sensor a
   top-(k+N) is computed once and sorted within (the exact top-k is
   recovered; routing intact) to obtain the runners-up and the
   top-k/top-k+1 margin.
2. `pending` receives `inds` (or `(inds, scores)` with `--ema-mass`).
3. `pos0 = lookup0[inds]`, `pos1 = lookup1[inds]`, masks `m0`, `m1`.
4. **Exact** (`mode == "exact"` or `T > 1` in `nosync/floor2d/absorb`):
   `pfx` from memmaps with the expert union (§2.4).
5. **Pool**: `y = gather_qmm(p0, s0·4, b_dyn, r0)·m0 + gather_qmm(p1, s1, 0,
   r1)·m1` per projection; GLU; down.
6. Depending on the mode: `floor2d` adds `gather_qmm(floor_w, floor_s·4, floor_b,
   inds)·(pos0 < 0)`; `floor` synchronously fetches the missing P0s; `absorb`
   transfers the dropped mass to the shared expert; `nosync` (with or without
   overflow) zeroes the gates of the absent experts and renormalizes.
7. Weighted sum + shared expert.

### The pager's `refresh_all()`

Per layer: drains `pending`, sanitizes indices, computes `counts` (by count
or by rescaled mass), measures the miss rate if needed (gears or selective
refresh), and calls `refresh` (or `refresh_fast` on the "fast" refreshes of
the overflow tier, when `--ovf-merge > 0`). Afterwards, if a gear
configuration exists, it evaluates the sensor (mean misses or 1 − normalized
margin) with hysteresis (`gear_hi_thr`, `gear_lo_thr`) and minimum dwell
(`gear_min_dwell = 2`) and shifts with `_shift_gear` (full rebuild of the
pools at the gear's (C, K)).

## 3.4 `serve.py` — the CLI

1. Reads `stream_layout` from the artifact's `config.json` and picks the
   runtime.
2. resident-p0: `mlx_lm.load`, `patch_fast`, optional governor. If the
   checkpoint has per-layer widths (Topiary taper), the lab's `dense_loader.
   maybe_patch_per_layer` (via `PYTHONPATH`) adapts it.
3. full-memmap: `pager.load_model`, `patch_pool(C, K, orders)`, mode and
   flags in `rt.S`, optional floor for `floor2d`.
4. Double prefill: `model(prompt)` + `refresh_all()`; then `stream_generate`
   with a refresh at the first token and every `--refresh` tokens (plus the
   burst if requested); the governor is consulted at every refresh.
5. Prints text, decode tok/s (excluding the first token) and peak memory
   (`mx.get_peak_memory()`).

## 3.5 Memory accounting

| Component | resident-p0 (35B) | full-memmap (80B, C=240, K=32) |
|---|---|---|
| Resident checkpoint | ~12 GB (P0 + non-expert) | skeleton ~4 GB |
| P0 pool | — | 240 × 3 proj × P0 row (~10 GB) |
| P1 pool | 32 × 3 × P1 row (~1–1.5 GB) | 32 × 3 × P1 row (~1.4 GB) |
| Universal floor | — | optional (80B 25%: 6 GB; does not fit with C=240) |
| Measured peak | 12.2–14.6 GB | 16.5–17.9 GB |

Operating rules: swap = 0 or the run is discarded; the peak is read from
`mx.get_peak_memory()` (the batteries store it in `peak_gb`). C=340 on the
80B (20.7 GB) does not fit; C=290 (19.8 GB) is the servable maximum.

## 3.6 mlx-lm dependencies we touch

- `mlx_lm.models.switch_layers._gather_sort` / `_scatter_unsort` (private
  helper; pending vendoring).
- Semantics of `gather_qmm(..., sorted_indices=True)`.
- `Qwen3MoeSparseMoeBlock` and `Qwen3NextSparseMoeBlock` constructors for
  the skeleton.
- CI and `pyproject` pinned to mlx 0.32.0 / mlx-lm 0.31.3 with bounded
  compatible ranges.
