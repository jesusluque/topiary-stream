# 4. Construir artefactos

Todo parte de un checkpoint 4-bit de `mlx-community` (cuantización afín,
`group_size 64`, `bits 4`; los routers ya vienen a 8 bits por override del
propio checkpoint). El entorno es el `.venv` del laboratorio
(`$LAB/.venv`, mlx 0.32.0 / mlx-lm 0.31.3) y los
comandos se lanzan desde la raíz de `topiary-stream`.

## 4.1 `split.py` — checkpoint → artefacto servible

```bash
# P0 cabe en RAM (35B, 30B): P0 en el checkpoint, P1 en memmaps
python src/split.py --src mlx-community/Qwen3.5-35B-A3B-4bit \
    --out artifacts/qwen35-stream --layout resident-p0

# P0 no cabe (80B, 235B): esqueleto + P0/P1/sb en memmaps; --consume borra
# cada shard fuente tras procesarlo (pico de disco ≈ checkpoint + 1 shard)
python src/split.py --src mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit \
    --out artifacts/qwen80-stream --layout full-memmap --consume
```

Qué hace por shard: carga con `mx.load`, detecta tensores de expertos por
nombre (`.switch_mlp.{gate,up,down}_proj.weight|scales|biases`), desempaqueta
los códigos 4-bit (`unpack4`), separa `codes >> 2` (P0) y `codes & 3` (P1),
los reempaqueta a 2 bits (`pack2`) y los escribe con `tofile` como
`uint32 [experts, words]`. En `full-memmap` además apila escalas y biases en
`float16 [experts, 2, sb_cols]` y sustituye cada bloque por tensores dummy de
`1×64`. Tiempo: ~15 min el 35B; el 80B ~1 h; el 235B varias horas (limitado
por disco).

Salida: directorio autocontenido con `config.json` (marcado con
`stream_layout` y, en full-memmap, `stream_skeleton`), tokenizer y plantilla
de chat, `model*.safetensors`, `p1_manifest.json` o `stream_manifest.json`,
y los `.bin`.

Para un checkpoint **Topiary** (anchos por capa, taper) el runtime necesita
el shim `dense_loader.maybe_patch_per_layer` del laboratorio:
`PYTHONPATH=$LAB/src`. `examples/build30stream.sh`
documenta la construcción completa del 30B-Stream (split + humo + suite).

## 4.2 `salience.py` — prior del pool y entrada del suelo

```bash
python src/salience.py --artifact artifacts/qwen80-stream \
    --data $LAB/data/calib_general_qwen3/calib.jsonl \
    --tokens 6000 --out artifacts/qwen80-stream/orders_routed.npz
```

Funciona **a través del pager en modo exacto**, sin checkpoint (en
full-memmap el original puede haber sido consumido). Espía el forward del
bloque: recalcula gate/up con los planos, acumula `E[h²]` por (capa,
experto, neurona) sobre el corpus, y al final multiplica por
`‖W_down[:, i]‖²` leído de los planos. Escribe `salience_{li}` de forma
`[E, inter]`. Coste ≈ 2× el forward exacto más una pasada de dequant por
(capa, experto). Consumidores: `patch_pool(..., orders=)` (prior de la EMA:
`salience.sum(axis=1)`) y `floor.py`.

Para leer la **curva de concentración** de un modelo (decide si el suelo
universal es viable): ordenar cada fila, acumular y mirar qué fracción de
energía captura el prefijo del 25 %/50 %. Medido: 80B 25 % → 61.6 %; 235B
16.7 % → 53.5 % (plano); modelos concentrados ~94 %.

## 4.3 `floor.py` — suelo universal 2D

```bash
python src/floor.py --artifact artifacts/qwen80-stream \
    --orders artifacts/qwen80-stream/orders_routed.npz \
    --k-floor 128 --out artifacts/qwen80-floor128.safetensors
```

Para cada capa toma las `k_floor` neuronas más salientes de cada experto
(`k_floor` múltiplo de 64), ordena el prefijo en orden natural (mantiene
coherente el recorte por columnas de `down`), extrae de los memmaps P0 las
filas (gate/up) o los grupos de columnas (down) y las escalas/biases
correspondientes, y guarda `L{li}.{proj}.{w,s,b}` y `L{li}.pref`. Tamaños:
80B `k=128` (25 % de anchura) 5.6 GB; 235B `k=256` (16.7 %) en el kit. Se
sirve con `--serve-mode floor2d --floor <fichero>`.

## 4.4 `pyramid.py` — pirámide anclada desde un máster 8-bit

```bash
python src/pyramid.py --stage anchored --src mlx-community/<modelo>-8bit --out models/m
#  → models/m-q4 (ancla), models/m-q8 (ancla+refinamiento), models/m-q2 (ancla truncada)
python src/pyramid.py --stage derive  --src <8bit> --bits 2 --out models/m-q2-derived   # control negativo
python src/pyramid.py --stage requant --src <8bit> --bits 2 --out models/m-q2-native    # baseline nativo
```

Exige máster **uniforme** de 8 bits sin overrides por módulo (lo comprueba).
Es la herramienta de la validación de §2.1; los artefactos de servicio
parten directamente del 4-bit de la comunidad (que es la ancla).

## 4.5 `protect.py` — protección de tensores por rangos HTTP (no medido)

```bash
python src/protect.py --artifact artifacts/qwen80-stream --router \
    --router-repo Qwen/Qwen3-Next-80B-A3B-Instruct          # routers a BF16 oficial
python src/protect.py --artifact artifacts/qwen80-stream --skeleton8 \
    --skel-repo mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit   # no-experto a 8 bits
```

Lee solo los tensores necesarios de los safetensors remotos (cabecera +
`Range GET`), escribe un **artefacto nuevo** (`qwen80-prot`, `qwen80-prot8`)
con los `.bin`/manifiestos enlazados simbólicamente y overrides por ruta en
`config.quantization` (`False` = sin cuantizar; `{"bits": 8}`). Nunca
modifica el artefacto de producción en sitio. Coste: +1.15 GB el esqueleto
a 8 bits. Quedó **sin medir** (campaña cancelada); corrección registrada: el
router ya iba a 8 bits, así que `--router` es 8→16, expectativa ≈ 0.

## 4.6 `awq_master.py` — máster protegido por saliencia (aparcado)

Registra una `AWQConfig` para `qwen3_moe` (atención tipo llama + escalas de
`switch_mlp` down/gate/up) en `mlx_lm.quant.awq` y delega en su `main`. Es la
única palanca no probada para la calidad del propio plano 2-bit (§6.6);
aparcada por disco (necesita el máster BF16) y por la regla de "sin rédito
claro, no".

## 4.7 Inventario de artefactos (2026-08-23)

| Artefacto | Layout | Tamaño | Origen | Notas |
|---|---|---|---|---|
| `artifacts/qwen30-stream` | resident-p0 | 13 GB | `qwen3-30b-topiary` (taper, checkpoint propio) | campeón en 9.2 GB; canales ordenados por saliencia |
| `artifacts/qwen35-stream` | resident-p0 | — | `mlx-community/Qwen3.5-35B-A3B-4bit` | movido a `/Volumes/Untitled/qwen35-stream-artifact` (sha256 verificado contra HF) |
| `artifacts/qwen80-stream` | full-memmap | 42 GB | `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` | + `orders_routed.npz`; 96 overrides en config (routers 8-bit) |
| `artifacts/qwen80-floor128.safetensors` | suelo 2D | 5.6 GB | floor.py k=128 | no cabe junto a C=240 |
| `artifacts/qwen80-prot`, `-prot8` | full-memmap (symlinks) | 1.3 / 2.4 GB | protect.py | sin medir |
| `artifacts/qwen235-stream` | full-memmap | 134 GB | `mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit-DWQ` (shards DWQ mezclados con hermanos 4-bit planos) | completo; P0 respaldado en `/Volumes/Untitled/qwen235-stream-p0` |
| `artifacts/qwen235-stream-kit` | esqueleto + floor256 + orders | 15 GB | — | lo que se sube a HF: el usuario regenera los planos con `split.py` |
| `artifacts/unsloth/…UD-Q2_K_XL.gguf` | GGUF (rival) | 28 GB | `unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF` | solo corre con `-ngl 0` en 24 GB |

Disco interno: ~16 GB libres tras mover el 35B al externo. Regla: las
decisiones de borrado son del usuario; el runtime nunca borra salvo
`--consume` explícito.

## 4.8 Publicación en Hugging Face

Repos (privados a fecha de hoy, pendientes de flip):
`jesusluque/qwen3.5-35b-topiary-stream`, `qwen3-next-80b-topiary-stream`,
`qwen3-235b-topiary-stream-kit`. El test `test_model_card_placeholder_gate`
impide subir cards con marcadores sin rellenar. Los checkpoints Topiary
(`qwen3-30b-topiary`, `-w640`, `-w576-code`) son públicos y sus cards ya
llevan la nota del precio del taper (−10 MMLU).
