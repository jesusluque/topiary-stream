# 2. Método

## 2.1 Pirámide de planos de bits anclada

### La identidad exacta

MLX cuantiza por grupos de `GROUP = 64` valores con la forma afín
`w = s·q + β` (escala `s` y bias `β` por grupo; `q ∈ [0, 15]` para 4 bits).
Escribiendo el código como `q4 = 4·q_hi + q_lo` con `q_hi, q_lo ∈ [0, 3]`:

```
w = s·(4·q_hi + q_lo) + β = [4s]·q_hi + [s]·q_lo + β
```

Es decir, un tensor 4-bit es la **suma de dos tensores 2-bit válidos**:

| Plano | Códigos | Escala | Bias |
|---|---|---|---|
| P0 | `q_hi = q4 >> 2` | `4·s` | `β` |
| P1 | `q_lo = q4 & 3` | `s` | `0` |

Cada uno se sirve con `mx.gather_qmm(..., bits=2, group_size=64)` tal como
viene en MLX. La reconstrucción P0+P1 a través del kernel es bit-exacta
(test `test_plane_reconstruction_exact_through_kernel`: error 2.6e-7, el del
redondeo float).

### El suelo: P0 solo con fold del centroide

Servir P0 sin P1 deja fuera `s·q_lo`. Con `q_lo` uniforme en {0,1,2,3} su
esperanza es `1.5·s`, así que el suelo usa bias `β + 1.5·s` (test
`test_cold_centroid_bias`). Es la constante insesgada a primer orden; medimos
que el plano descartado es casi uniforme en la práctica (media global 1.504)
y que sustituirla por la media empírica por grupo (`--centroid empirical`)
solo mejora el MSE del suelo un 1.5 % — negativo documentado (§7).

### Dirección de construcción (lo que costó una inversión aprender)

Se puede derivar una pirámide **truncando hacia abajo** un máster fino (Q8 →
Q4 → Q2 por desplazamiento de bits) o **anclándola** en el nivel de servicio.
La truncación gana en todas las métricas uniformes de peso (L2 y error
máximo) y pierde de extremo a extremo: en OLMoE-7B el Q2 derivado fue ~20×
peor en PPL que un Q2 ajustado nativamente. El grid truncado concentra su
error en los pesos salientes, que las métricas uniformes infraponderan.

Regla adoptada (`pyramid.py --stage anchored`):

| Nivel | Construcción | Calidad |
|---|---|---|
| Q4 | ajuste min/max nativo desde el máster 8-bit | calidad de servicio exacta |
| Q8 | ancla Q4 + plano de refinamiento de 4 bits (`codes8 = q4·16 + q_lo4`; `>>4` recupera Q4 exactamente) | PPL 3.9209 vs máster verdadero 3.9213: gratis |
| Q2 | ancla truncada (`q4 >> 2`, `s·4`, `β + 1.5·s`) | suelo, **solo** para slots fríos protegidos por gate |

`--stage derive` (truncación ingenua) y `--stage requant` (reajuste nativo a
N bits) se conservan como constructores del control negativo y del baseline.

### Alcance del formato

La partición exacta exige cuantización **afín plana** (`s·q + β` por grupo).
No porta a los formatos fuertes de GGUF: los K-quants anidan escalas de 6
bits en superbloques y los IQ-quants son de codebook. Solo el afín por
bloque tipo Q4_1 es estructuralmente compatible. Es la cara B de la
propiedad "kernels de serie" y se declara como tal. (Hacia arriba sí porta:
Q8_0 → 2×Q4_0 es simétrico sin fold.)

## 2.2 Artefactos: dos layouts

`split.py` convierte un checkpoint 4-bit de la comunidad (`mlx-community/…`)
en un directorio autocontenido (config, tokenizer, planos) que es lo que se
sube, se descarga y se sirve.

### `resident-p0` (P0 cabe en RAM: 35B, 30B)

- Los tensores de expertos (`*.switch_mlp.{gate,up,down}_proj.weight`) se
  sustituyen en el checkpoint por su plano P0 (2 bits); el `config.json`
  gana overrides por módulo `{"group_size": 64, "bits": 2}` y la marca
  `stream_layout = "resident-p0"`.
- P1 va a memmaps `L{li}.{proj}.p1.bin` (`uint32`, `[experts, words]`) con
  `p1_manifest.json` (experts, out, cols, words, layer, proj).
- Escalas y biases se quedan en el checkpoint (son del 4-bit original; el
  runtime multiplica `s·4` para P0).
- RAM al cargar ≈ 60 % del modelo 4-bit (35B: 19.5 → ~12 GB).

### `full-memmap` (P0 no cabe: 80B, 235B)

- Nada con forma de experto queda en el checkpoint: `L{li}.{proj}.p0.bin`,
  `.p1.bin` y `.sb.bin` (escalas y biases apilados `float16`,
  `[experts, 2, sb_cols]`) paginables por fila de experto.
- Un **esqueleto** (`model.safetensors`, 3–5 GB) con los pesos no-experto y
  tensores switch *dummy* de 1×64; `config.json` lleva `stream_skeleton =
  true`. Al cargar, `pager.maybe_patch_skeleton` parchea el constructor del
  bloque MoE (`Qwen3MoeSparseMoeBlock`, `Qwen3NextSparseMoeBlock`) para que
  instancie `SwitchGLU(hidden, 64, 1)` y el checkpoint cuadre.
- `stream_manifest.json` describe cada `L{li}.{proj}`: experts, out, cols,
  words, sb_cols, s_shape.

`--consume` borra cada shard fuente (enlace y blob de la caché HF) tras
procesarlo, acotando el pico de disco a ~checkpoint + un shard de planos. El
esqueleto se escribe **primero**, así el runtime nunca necesita la fuente
consumida — lección pagada con una re-descarga de 132 GB.

## 2.3 Pool gobernado por el gate, sin sincronización

### La ventaja estructural

En un bloque MoE la salida es `y = shared(x) + Σ_{i∈topk} g_i·E_i(x)`. El
router (`gate`) es una matriz diminuta que se evalúa **antes** de leer los
expertos: la "distancia de cámara" se conoce a priori, gratis. Stream usa
esa información para decidir qué expertos están residentes y en qué
precisión, sin predecir nada.

### Política de pertenencia, no umbral por token

Por capa, el pool mantiene:

- resident-p0: K expertos con su plano P1 en GPU (`pool [K, out, in_words]`,
  `pool_s [K, …]`); P0 de todos está residente en el checkpoint.
- full-memmap: C expertos a P0 (`p0 [C, …]`, `s0`, `b_dyn`) y K de ellos
  con P1 (`p1 [K, …]`, `s1`).

La pertenencia va codificada en un **tensor de biases dinámico** `b_dyn`:
los miembros con P1 llevan `β` (su P1 los completa a 4-bit exacto); el resto
`β + 1.5·s` (suelo). Así **una** llamada `gather_qmm` sobre P0 sirve
calientes y fríos, y una segunda sobre el pool P1 (con índices remapeados y
máscara de pertenencia) añade el detalle a quien lo tiene. No hay `if` por
token ni copia CPU↔GPU por capa.

Esto fue la diferencia entre 8.1 y 44.5 tok/s en el 35B: la versión de
referencia sincronizaba ~40 veces por token (una por capa para consultar
pertenencia); la ruta rápida no sincroniza ninguna.

### Refresh: la única sincronización diferida

En el forward, los índices top-k se apilan en `pending[layer]` (lazy, sin
evaluar en decode). Cada `REFRESH` tokens (256 por defecto; 128 en las
baterías con `--gen-refresh 128`), `refresh_all()`:

1. Concatena y evalúa los índices pendientes, los sanea (`0 ≤ idx < E`:
   blindaje contra índices basura bajo presión de memoria).
2. Actualiza la EMA por capa: `ema = α·ema + counts` (α = 0.7 resident-p0,
   0.8 full-memmap) — frecuencia + recencia, la señal que predice la
   localidad del MoE.
3. Recalcula el top-C / top-K y pagina **solo las filas entrantes** desde el
   memmap (`np.memmap` lee únicamente las páginas tocadas).

En resident-p0 el pool se repagina entero ante cualquier cambio (K=32 filas
completas son ~40 MB desde page cache; refresh=64 costó solo −2.5 % tok/s).
En full-memmap las filas son ~10× mayores y la reconstrucción completa costó
10× en decode (~13 GB de copias cada 64 tokens): el refresh es
**incremental**, con churn acotado (`MAX_CHURN = 8` entrantes por refresh,
que desplazan a los 8 de menor EMA) y solo el diff de la rotación de P1 se
toca. `mx` `setitem` copia el buffer entero, de ahí la obsesión por tocar
poco (~1.7 s por refresh completo en el 80B).

### Un miss nunca bloquea

Si un experto enrutado no está en el pool:

- `nosync` (producción): se elimina y los gates restantes se renormalizan
  (*drop-renormalize*). Rápido; el daño crece con la dispersión del dominio.
- `floor`: se trae su P0 del memmap de forma síncrona. Correcto pero ligado
  a la sincronización: ~0.2 tok/s con 94 capas. Modo de demostración.
- `floor2d`: se sirve su rebanada del suelo universal residente (§2.5).
- `absorb`: el experto compartido absorbe la masa caída. **Negativo
  rotundo** (KLD 7.17): las escalas de salida no son comparables.
- `exact`: todo slot se sirve P0+P1 directo del memmap (== 4-bit verdadero);
  solo batch/teacher-forced (~4 GB de pico). Es como se mide la base
  verdadera de un modelo que no cabe.

## 2.4 Prefill exacto

Cualquier forward con `T > 1` (el prompt, o un chunk teacher-forced) es una
pasada batcheada: se lee P0 como siempre y el P1 de la **unión** de expertos
que el prompt necesita se trae una sola vez del memmap
(`np.unique(inds)` → `rme` índices remapeados → dos `gather_qmm`). La
política del pool gobierna, por tanto, solo el decode.

Es la decisión de diseño que borró el mayor peaje medido del runtime: el
prefill servido por pool costaba +6–11 % de PPL en el 35B y +28 %/+120 % en
el 80B (el routing de prefill es plano y derrota a la pertenencia por
recencia), y dejaba al decode una caché KV envenenada. Con prefill exacto:
PPL servida = base a cuatro decimales, KLD teacher-forced = 0.000, y el 80B
pasó de 84 %/96 % a 15/15 / 15/15 en HumanEval/GSM8K.

`serve.py` además hace **doble prefill**: una pasada del prompt solo para
estadísticas de routing (el pool se adapta), y la pasada real contra un
pool informado. Sin ello un pool uniforme frío servía al prompt degradado
(medido en el 235B).

## 2.5 Suelo universal 2D (`floor.py` + `floor2d`)

Cada experto aporta sus `k_floor` neuronas más salientes (por *orders*
enrutados) al nivel P0, empaquetadas en un `safetensors` residente. Con el
suelo cargado ningún slot se cae nunca: los expertos fuera del pool sirven su
rebanada. Para `down_proj` el recorte es por grupos de 64 columnas de
entrada (coherente con la cuantización por grupos); para `gate/up` por filas
de salida.

Guía honesta medida: la calidad del suelo sigue a la **concentración de la
saliencia**. Con el prefijo capturando >85–90 % de la energía es un nivel de
calidad real (30B-Stream con P0 de todos los expertos: KLD 0.131); a ~50 %
(235B: 53.5 %; 80B al 25 % de anchura: 61.6 %) degenera a coherencia de
último recurso y puede no ser servible (235B: texto degenerado; 80B: 1.354 a
C=120 y no cabe junto al pool de C=240 → thrashing).

## 2.6 Gobernador elástico

La residencia es un dial, así que lleva un controlador. En cada refresh,
`available_gb()` lee `vm_stat` (libres + inactivas + purgables +
especulativas × tamaño de página) y `govern(low=4, high=7, k_min=4,
k_max=48, step=8)` encoge K si `avail < low` o lo crece si `avail > high`;
el siguiente refresh materializa el cambio (los pools se reconstruyen a
`pool_k` por diseño). Medido bajo un globo externo de 8 GB
(`examples/balloon.py`): cuatro repliegues automáticos (K 32→4, memoria
activa 13.7→12.8 GB) con la generación completándose limpia; y con el
gobernador activo el 35B rindió 51.1 tok/s (probablemente encogió K solo).

## 2.7 Variantes implementadas

**Medidas (y con veredicto):**

- `--p1-frac g,u,d`: servir P1 solo para el prefijo saliente de cada
  proyección (rebanada contigua porque los checkpoints Topiary traen los
  canales ordenados por saliencia; `0` = proyección a puro suelo). Veredicto:
  dial de emergencia de memoria, no optimización general (§6.3).
- `--centroid empirical`: negativo (1.5 % MSE).
- Cadencia de refresh (`--refresh`, `--gen-refresh`, `--kld-refresh`): LA
  palanca de KLD, no mueve tareas; coste en tok/s (§6.5).
- Tier de desbordamiento (`--ovf-merge N`): negativo (§7).
- Modo 2-bit / segunda marcha (C alto, K≈1): negativo en tareas, −25 % KLD
  en prosa (§6.5, §7).

**Implementadas como flags pero NO medidas** (la campaña se canceló antes;
no se reclama nada de ellas): ráfaga de refresh tras el prompt
(`--burst-len/--burst-every`), gobernador de dos marchas (`--gear`,
`--gear-hi/lo`, sensor `miss`/`margin`), EMA ponderada por masa de gate
(`--ema-mass`), precalentamiento del desbordamiento con casi-elegidos
(`--prewarm N`), refresh selectivo por capa (`--refresh-min-miss`),
protección de tensores (`protect.py`: router a BF16, esqueleto a 8 bits).
Siguen en el código porque son pequeñas y aisladas; su valor por defecto
las desactiva.
