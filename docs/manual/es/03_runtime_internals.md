# 3. El runtime por dentro

Recorrido por `src/` en el orden en que se ejecuta. Nada de esto depende de
nombres de módulos concretos del modelo: los bloques MoE se descubren por
contenido.

## 3.1 `common.py` — helpers compartidos

- `set_seeds(1234)`: `mx.random.seed` + `np.random.seed`. Con decodificación
  greedy las semillas solo afectan a la selección de muestras de los
  benchmarks.
- `load_corpus(path, limit_tokens)`: lee `jsonl` con `{"text", "n_tokens"}`
  hasta un presupuesto de tokens.
- `token_nll(logits, ids)`: NLL teacher-forced por token con log-softmax
  directo `z − logsumexp(z)` (estable con vocabularios de 248k; `log(softmax
  + eps)` perdía las colas).
- **Descubrimiento de bloques MoE** (`find_moe_blocks`): recorre
  `model.layers` (desenvolviendo `language_model` si existe) y acepta como
  bloque cualquier módulo — o la propia capa — que tenga un hijo router
  (`gate|router|gate_proj|wg`) y un contenedor de expertos
  (`switch_mlp|experts|mlp|moe`) con al menos un tensor 3-D
  `[n_experts, out, in]`. Devuelve `[(layer_index, block)]`.
- **Aritmética de planos**: `unpack4` (uint32 empaquetado → códigos 0..15;
  MLX empaqueta little-endian, nibble bajo primero), `pack2` (códigos 2-bit →
  uint32, 16 por palabra), `pack`/`unpack_bits` genéricos para 2/4/8 bits
  (tests `test_unpack_pack_roundtrip`, `test_unpack_bits_order`).

## 3.2 `fastpath.py` — runtime `resident-p0`

### Estado por capa: `FastLayer`

Construido en `patch_fast` para cada bloque MoE con su entrada del
`p1_manifest.json`:

| Campo | Contenido |
|---|---|
| `projs[name]` | por proyección: `p0` (el `weight` del checkpoint, ya 2-bit), `s`, `b`, `b_cold = b + 1.5·s`, `mm` (memmap P1), `out`, `b_dyn`, `r` (filas/palabras de P1 servidas), `pool`, `pool_s` |
| `pool_k` | K actual (el gobernador lo cambia en vivo) |
| `lookup` | `int32[E]`: posición en el pool o −1 |
| `members` | lista de expertos con P1 |
| `ema` | `float64[E]` |

Con `--p1-frac`, `r` recorta: gate/up por filas de salida (múltiplo de 64);
down por grupos de palabras de entrada (`rg·4` palabras = `rg` grupos de
64). `frac = 0` deja `r = 0` y la proyección vive solo del suelo.

### Forward parcheado

`patch_fast` sustituye `block_cls.__call__`:

1. `gates = softmax(gate(x))` en float32 (`precise=True`); `inds =
   argpartition` top-k (sin ordenar — el mismo conjunto que el original);
   `scores` normalizados si `norm_topk_prob`.
2. En prefill (`x_flat.shape[0] > 1`) se **evalúa `inds` ya** (índices lazy
   bajo presión de memoria produjeron basura) y se sirve exacto (§2.4):
   `pfx` = `gather_qmm(P0, s·4, b)` + `gather_qmm(P1_unión, s_u, 0)`.
3. En decode: `inds` se apila en `STATE["pending"]` **sin evaluar**; si
   `inds.size ≥ 64` se usa `switch_layers._gather_sort` (ordena tokens por
   experto para el kernel con `sorted_indices=True`) y `_scatter_unsort` a la
   salida. Por proyección: `y = gather_qmm(P0, s·4, b_dyn, idx)` +
   `gather_qmm(pool, pool_s, 0, remap)·mask`, donde `remap = max(lookup[idx],
   0)` y `mask = lookup[idx] ≥ 0`. Con `r < out` en gate/up se rellena con
   `mx.pad`; con down se recorta la entrada `xin[..., :r·16]`.
4. Activación GLU del bloque original, suma ponderada por `scores`, más el
   experto compartido si existe (`sigmoid(shared_expert_gate(x)) ·
   shared_expert(x)`).

### Refresh y gobernador

- `refresh_all()`: por capa, concatena los índices pendientes, los sanea y
  llama a `FastLayer.refresh(counts)`: `ema = 0.7·ema + counts`, `want =
  top-K`; si no entra nadie nuevo, nada que hacer; si entra alguien, repagina
  el pool entero desde el memmap (filas pequeñas), recalcula `pool_s`,
  `lookup` y `b_dyn = b_cold` con `b` en los miembros. Devuelve el número de
  entrantes.
- `available_gb()` y `govern(...)`: ver §2.6. Devuelve un mensaje
  `[gov] avail X GB -> K a->b` cuando cambia y vacía `members` para forzar la
  reconstrucción al siguiente refresh.

## 3.3 `pager.py` — runtime `full-memmap`

### Carga

`load_model(artifact)`: si `config.stream_skeleton`, parchea los
constructores (`maybe_patch_skeleton`) y carga el esqueleto entero; si no,
carga lazy y evalúa todo menos los tensores switch (camino de modelos sin
esqueleto).

### Estado por capa: `PoolLayer`

| Campo | Contenido |
|---|---|
| `mm[proj]` | memmaps `p0`, `p1` (`uint32 [E, words]`), `sb` (`float16 [E, 2, sb_cols]`), `out`, `s_shape` |
| `c`, `k` | tamaños del pool |
| `ema` | inicializada con el **prior** (`orders_routed.npz`: `salience_li.sum(axis=1)`) o uniforme |
| `members0/1`, `lookup0/1` | pertenencia a P0 / P1 y posiciones |
| `pools[proj]` | `p0 [C,…]`, `s0`, `b_dyn`, `p1 [K,…]`, `s1`; más el tier de desbordamiento `po/so/bo [OVF=32,…]` |

`_install(want0)` hace la reconstrucción completa (solo en init y en cambio
de marcha); `refresh(counts)` es el incremental de §2.3. `_rows(proj, plane,
ids)` y `_sb(proj, ids)` leen filas del memmap y las convierten a `mx.array`
con la forma `[n, out, -1]`.

### Forward parcheado, por modo

1. Routing como en fastpath. Con `--prewarm N` o sensor `margin` se calcula
   un top-(k+N) una sola vez y se ordena dentro (el top-k exacto se
   recupera; routing intacto) para obtener los casi-elegidos y el margen
   top-k/top-k+1.
2. `pending` recibe `inds` (o `(inds, scores)` con `--ema-mass`).
3. `pos0 = lookup0[inds]`, `pos1 = lookup1[inds]`, máscaras `m0`, `m1`.
4. **Exacto** (`mode == "exact"` o `T > 1` en `nosync/floor2d/absorb`):
   `pfx` desde memmaps con la unión de expertos (§2.4).
5. **Pool**: `y = gather_qmm(p0, s0·4, b_dyn, r0)·m0 + gather_qmm(p1, s1, 0,
   r1)·m1` por proyección; GLU; down.
6. Según el modo: `floor2d` añade `gather_qmm(floor_w, floor_s·4, floor_b,
   inds)·(pos0 < 0)`; `floor` trae síncronamente los P0 que faltan; `absorb`
   transfiere la masa caída al experto compartido; `nosync` (con o sin
   desbordamiento) pone a cero los gates de los ausentes y renormaliza.
7. Suma ponderada + experto compartido.

### `refresh_all()` del pager

Por capa: vacía `pending`, sanea índices, calcula `counts` (por cuenta o por
masa reescalada), mide la tasa de misses si hace falta (marchas o refresh
selectivo), y llama a `refresh` (o `refresh_fast` en los refreshes
"rápidos" del tier de desbordamiento, cuando `--ovf-merge > 0`). Después, si
hay configuración de marchas, evalúa el sensor (misses medios o 1 − margen
normalizado) con histéresis (`gear_hi_thr`, `gear_lo_thr`) y permanencia
mínima (`gear_min_dwell = 2`) y conmuta con `_shift_gear` (reconstrucción
completa de pools a la (C, K) de la marcha).

## 3.4 `serve.py` — el CLI

1. Lee `stream_layout` del `config.json` del artefacto y elige runtime.
2. resident-p0: `mlx_lm.load`, `patch_fast`, gobernador opcional. Si el
   checkpoint tiene anchos por capa (taper de Topiary), `dense_loader.
   maybe_patch_per_layer` del laboratorio (vía `PYTHONPATH`) lo adapta.
3. full-memmap: `pager.load_model`, `patch_pool(C, K, orders)`, modo y
   flags en `rt.S`, suelo opcional para `floor2d`.
4. Doble prefill: `model(prompt)` + `refresh_all()`; luego `stream_generate`
   con refresh al primer token y cada `--refresh` (más la ráfaga si se
   pide); el gobernador se consulta en cada refresh.
5. Imprime texto, tok/s de decode (sin el primer token) y pico de memoria
   (`mx.get_peak_memory()`).

## 3.5 Contabilidad de memoria

| Componente | resident-p0 (35B) | full-memmap (80B, C=240, K=32) |
|---|---|---|
| Checkpoint residente | ~12 GB (P0 + no-experto) | esqueleto ~4 GB |
| Pool P0 | — | 240 × 3 proy × fila P0 (~10 GB) |
| Pool P1 | 32 × 3 × fila P1 (~1–1.5 GB) | 32 × 3 × fila P1 (~1.4 GB) |
| Suelo universal | — | opcional (80B 25 %: 6 GB; no cabe con C=240) |
| Pico medido | 12.2–14.6 GB | 16.5–17.9 GB |

Reglas operativas: swap = 0 o el run se descarta; el pico se lee de
`mx.get_peak_memory()` (las baterías lo guardan en `peak_gb`). C=340 en el
80B (20.7 GB) no cabe; C=290 (19.8 GB) es el máximo servible.

## 3.6 Dependencias de mlx-lm que tocamos

- `mlx_lm.models.switch_layers._gather_sort` / `_scatter_unsort` (helper
  privado; pendiente de vendorizar).
- Semántica de `gather_qmm(..., sorted_indices=True)`.
- Constructores `Qwen3MoeSparseMoeBlock` y `Qwen3NextSparseMoeBlock` para el
  esqueleto.
- CI y `pyproject` fijados a mlx 0.32.0 / mlx-lm 0.31.3 con rangos
  compatibles acotados.
