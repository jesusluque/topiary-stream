# 1. Visión y lineaje

## 1.1 Qué problema resuelve

Un MoE cuantizado a 4 bits que no cabe en la memoria unificada de un
portátil no se puede servir: el cargador lo trae entero o no lo trae. En un
MacBook Pro M5 Pro de 24 GB eso deja fuera al Qwen3.5-35B (19.5 GB: carga,
pero Metal muere al generar), al Qwen3-Next-80B (42 GB) y al Qwen3-235B
(132 GB).

Topiary Stream es un **runtime de residencia**: el modelo completo vive en
disco en un formato paginable por experto, y en RAM solo están (a) el
esqueleto no-experto y (b) un *pool* por capa con los expertos que el
router está usando, en la precisión que el router justifica. El resultado,
medido en esa máquina:

| Modelo | Tamaño 4-bit | Pico servido | tok/s | Calidad |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | 19.5 GB | 12.5–14 GB | 44.5–47 | PPL = base; HumanEval/GSM8K 92/92 (n=25/50) |
| Qwen3-Next-80B-A3B | 42 GB | 17–17.6 GB | 17–21.5 | PPL = base; MMLU 85.2 (n=500); MATH-500 64–65, MBPP 81 (n=100) |
| Qwen3-235B-A22B | 132 GB | — | — | sin término medio servible a 24 GB (negativo mapeado) |

Todo con **kernels cuantizados de serie de MLX** (`gather_qmm`): sin Metal
propio, sin motor de inferencia dedicado.

## 1.2 Lineaje

El proyecto tiene tres capas, cada una con su repositorio:

1. **nanite-moe** (laboratorio, privado). Nació como "LOD jerárquico de
   expertos por gate" inspirado en Nanite: cámara = estado oculto, distancia
   = score del router, LOD = precisión con la que se lee el experto. Fases
   F0–F48 con informes numerados en `reports/` (arquitecturas, localidad del
   routing, calidad dual-precisión, eje de rango, anchura, anidado,
   clusters, composición, escala, runtime, suite final, prior art,
   auditoría, streaming simulado, confrontaciones externas). Allí se
   midieron las leyes que Stream explota: localidad del routing en decode,
   descomposición exacta de un tensor 4-bit en dos planos de 2 bits,
   dirección correcta de construcción de la pirámide, persistencia
   token-a-token del routing (f48).
2. **Topiary** (público: código y checkpoints `qwen3-30b-topiary*` en HF).
   Esculpido estático por saliencia: poda de anchura de expertos con
   canales reordenados por saliencia (el "taper"). Produce checkpoints más
   pequeños y algo más rápidos; su precio, medido después, es conocimiento
   (−10 MMLU) y no razonamiento.
3. **Topiary Stream** (este repositorio; privado a fecha de hoy). El
   runtime de residencia. Hereda de Topiary el orden de canales por
   saliencia (lo que hace contiguo el "prefijo saliente" del plano P1) y del
   laboratorio la pirámide anclada y la telemetría del router.

## 1.3 La tesis, y su corrección

Tesis inicial: *la asignación temporal de bits guiada por el gate domina al
downsizing estático a igualdad de memoria*. Medida contra el mejor estático
disponible (Unsloth UD-Q2_K_XL del mismo 80B, imatrix, ~3 bpw, 30 GB), la
tesis queda **corregida**:

- En tareas, el estático calibrado está a la par o nominalmente por encima
  (MATH 69 vs 65, MBPP 86 vs 81, MMLU 86.2 vs 85.2; n=100/100/500).
- En fidelidad de texto general (KLD contra la base exacta) el estático gana
  con claridad (0.195 vs 0.774 en producción).
- Lo que Stream aporta es **de sistema**: cabe donde el estático no cabe
  (17 vs 30 GB), es un 40–70 % más rápido (GPU frente a CPU paginando),
  sirve el prompt bit-exacto, tiene suelo garantizado y gobernador elástico.

La formulación honesta: *la asignación temporal de bits es la forma de
correr bien un modelo de 42 GB en 24 GB; no es una forma de superar en
calidad a un estático bien calibrado que necesita 30.*

## 1.4 Las tres ideas, en una frase cada una

1. **Pirámide de planos de bits anclada.** Un tensor afín de 4 bits es
   exactamente dos tensores de 2 bits (`q4 = 4·q_hi + q_lo`), ambos válidos
   para el kernel de serie; P0 (bits altos, con sesgo de centroide) es un
   suelo servible a la mitad de bytes; P0+P1 es el 4-bit bit-exacto.
2. **El gate gobierna la residencia.** Los scores del router existen antes
   de leer un solo byte de experto; la decisión de residencia es gratis. La
   pertenencia al pool va codificada en un tensor de *biases* dinámico, así
   que una sola llamada al kernel sirve calientes y fríos sin sincronizar
   CPU/GPU por token.
3. **Respira.** El gobernador lee la presión real de memoria de macOS en
   cada refresh y redimensiona el pool; un miss nunca bloquea (sirve el
   suelo "borroso un fotograma") y el pool lo aprende en el siguiente
   refresh.

## 1.5 Glosario

| Término | Significado |
|---|---|
| **P0 / P1** | Planos de bits altos / bajos de un código de 4 bits. P0 solo = nivel 2-bit (suelo); P0+P1 = 4-bit exacto. |
| **Pirámide anclada** | Q4 nativo como ancla; Q8 = ancla + plano de refinamiento; Q2 = ancla truncada (solo para slots fríos protegidos por gate). |
| **Fold del centroide (1.5·s)** | Constante sumada al bias cuando se sirve P0 solo: esperanza del plano descartado bajo q_lo uniforme. Medido casi uniforme (media 1.504). |
| **resident-p0** | Layout de artefacto en el que P0 vive en el checkpoint y P1 en memmaps por (capa, proyección). Para modelos cuyo P0 cabe en RAM (35B, 30B). |
| **full-memmap** | Layout en el que P0, P1 y escalas/biases van todos a memmaps por fila de experto; el checkpoint es un esqueleto de 3–5 GB. Para 80B y 235B. |
| **Pool** | Conjunto residente por capa: K expertos con P1 (resident-p0) o C expertos a P0 de los cuales K con P1 (full-memmap). |
| **C / K** | Tamaño del pool a P0 / número de expertos con detalle P1. Producción 80B: C=240, K=32 (de 512 expertos/capa). |
| **Refresh** | La única sincronización diferida: vacía los contadores de routing en una EMA, recalcula la pertenencia y pagina las filas entrantes. Cadencia por defecto 256 tokens. |
| **Miss / drop** | Experto enrutado que no está en el pool. En `nosync` se elimina y se renormalizan los gates restantes (drop); en `floor*` se sirve su suelo. |
| **Suelo universal (floor2d)** | P0 × prefijo saliente de todos los expertos, residente, para que ningún slot se caiga nunca. Viable si la saliencia está concentrada (≥85–90 % capturado). |
| **Prefill exacto** | Cualquier forward con T>1 se sirve a 4-bit completo (P1 de la unión de expertos leído una vez del memmap). La política de pool solo gobierna el decode. |
| **Gobernador** | Lazo presión→K: lee `vm_stat` en cada refresh y encoge/crece K entre cotas. |
| **Taper** | Poda de anchura por saliencia de Topiary (canales reordenados). |
| **Orders / saliencia enrutada** | `E[h²]·‖W_down[:,i]‖²` por (capa, experto, neurona), calculada a través del pager sin checkpoint; prior del pool y entrada del suelo. |
| **KLD en decode** | KL(base ‖ servido) token a token con caché KV viva, prefijo exacto de 16 tokens, refresh a cadencia de producción. La métrica hostil del peaje del pool. |
| **Ley de cobertura** | Relación medida entre fracción de expertos residentes y daño: 23 % rompe razonamiento largo; 47 % sirve; 100 % a P0 (suelo universal) reduce la KLD 6×. |
