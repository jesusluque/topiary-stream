# Banco de soluciones — 2026-08-19

Batería idéntica para todas las configuraciones: MATH-500 (n=100, estratificado
por nivel), MBPP sanitized (n=100), MMLU (n=500, estratificado por materia),
LAMBADA (n=500). Greedy, seed 1234, mismas muestras. M5 Pro 24 GB, MLX.
Registros: `runs/bench_*.json`, logs en `runs/`.

## 1. Calidad — cinco modelos, cuatro ejes

| Bench | 30B original | 30B Topiary | 30B-Stream | 35B Stream | 80B Stream |
|---|---|---|---|---|---|
| RAM servida | 16.4–17.9 GB | 14.5–15.2 GB | **9.2–11.3 GB** | 12.5–14.6 GB | 16.5–17.9 GB |
| MATH-500 | 70% | **72%** | 67% | 60% | 64% |
| MBPP | **83%** | 81% | 82% | 78% | 81% |
| MMLU | 78.2% | 68.2% | 69.4% | 83.0% | **85.2%** |
| LAMBADA | 64.6% | 60.2% | 60.0% | 75.4% | **78.2%** |

**Hallazgo principal (el precio real del taper):** la poda por saliencia del
30B preservó el razonamiento por completo (MATH 72 vs 70, MBPP 81 vs 83 — n.s.
ambas) pero **costó −10 puntos de MMLU** (68.2 vs 78.2; 50 ítems a n=500, muy
significativo) y −4.4 de LAMBADA. El taper es un recorte de *conocimiento*, no
de *razonamiento*. Hasta hoy era invisible porque solo se evaluaba con
benchmarks de razonamiento. HECHO el mismo día: README de topiary, paper §3.7
y las 3 cards públicas de HF actualizadas (commit topiary 032b039).

**Especialización por solución:** el esculpido domina razonamiento; los Stream
grandes dominan conocimiento (MMLU/LAMBADA +15–18 sobre los 30B). La razón de
paginar modelos que no caben es exactamente el conocimiento que solo la masa
total de parámetros tiene. Caveat: las columnas mezclan familias (Qwen3 /
Qwen3.5 / Next) — técnica y modelo no están separados salvo en el par
original/taper (mismo modelo) y taper/30B-Stream (mismo checkpoint).

**El 30B-Stream valida el runtime a coste ~cero:** vs su checkpoint fuente
(taper) queda a −5 MATH (borde de significancia), +1 MBPP, +6 ítems MMLU, −1
LAMBADA — indistinguible en 3 de 4 ejes con **5.3 GB menos** (9.2 vs 14.5).
El campeón cabe ahora en máquinas de 16 GB.

## 2. Velocidad — matriz de configuraciones (256 tok, máquina caliente)

| Config | tok/s | Pico GB |
|---|---|---|
| 30B original nativo | 101.4–103.0 | 17.2 |
| 30B Topiary nativo | **108.1–108.3** | **14.5** |
| 30B-Stream K=32 | 47.9 | 11.1 |
| 35B K=4 | 54.8 | 12.9 |
| 35B K=32 (producción) | 47.6 | 13.9 |
| 35B K=64 | 44.5 | 15.0 |
| 35B refresh=64 | 46.4 | 13.9 |
| 35B + governor | 51.1 | 13.9 |
| 80B C=240 (producción) | 17.3 | 17.0 |
| 80B C=120 | 28.1 | 10.2 |
| 80B refresh=64 | 12.4 | 17.0 |

Remedido mismo día, rondas intercaladas, solo-decode (examples/speed_nativo.py):
el taper compra **+6% tok/s y −2.7 GB** — real pero modesto. La reducción de
bytes es 16% y la ganancia 6%: el 30B a batch-1 no es puramente memory-bound
en esta máquina. (El "87 vs 104" previo era un artefacto de comparar sesiones.)

Notas: el governor sale gratis (51.1 — probablemente encogió K solo). El 80B
C=120 (+62% velocidad, −40% memoria) queda **sin validar en calidad** —
cobertura 23%, zona de riesgo; no adoptar sin su batería.

## 3. Subsampling saliente del plano P1 (4:2:2) — curva completa

Sobre el 30B-Stream (canales ya ordenados por saliencia de fábrica en
checkpoints topiary). Notación (gate:up:down), 4=P1 completo, 2=50% saliente,
1=25%, 0=sin P1 (puro suelo P0). K=32 salvo indicado.

| Config | Bytes pool | MATH-500 | MBPP | MMLU |
|---|---|---|---|---|
| 4:4:4 (baseline) | 3.0× | 67% | 82% | 69.4% |
| 4:4:2 | 2.5× | **72%** | — | — |
| 2:2:4 | 2.0× | 69% | **74%** | **65.8%** |
| 2:4:2 | 2.0× | 65% | — | — |
| 2:2:2 (K=64, mismos bytes que baseline) | 3.0× | 66% | — | — |
| 2:2:2 | 1.5× | 65% | — | — |
| 1:1:4 | ~1.2× | 64% | — | — |
| 0:0:4 | ~1.0× | 64% | — | — |

**Conclusiones:**
1. La curva MATH-vs-bytes es casi plana: de 3.0× a 1.0× solo se pierden 3
   ítems. La matemática tolera el suelo de 2 bits en gate/up.
2. **Pero no generaliza entre ejes**: la batería completa del 2:2:4 mostró
   −8 MBPP y −3.6 MMLU vs baseline. El recorte saliente NO es gratis.
3. Los "ganadores" de MATH (4:4:2 = 72, 2:2:4 = 69 > baseline 67) son
   sospechosos de ruido a n=100 (mejorar quitando detalle no tiene mecanismo);
   el rango entero 64–72 requiere n mayor o McNemar pareado para ordenarse.
4. **Default adoptado: P1 completo (4:4:4).** El recorte saliente queda como
   dial de emergencia de memoria (0:0:4 sirve MATH decente a 10.4 GB de pico),
   no como optimización general.
5. La hermana barata (centroide empírico por grupo, `--centroid empirical`)
   se descartó sin GPU: los 2 bits descartados son ruido casi uniforme
   (mejora MSE 1.5%). Y la idea externa de delta-compression entre filas se
   falsó con medición directa: igualdad entre filas vecinas = tasa de azar
   exacta (8.6% en q4) → el esquema máscara+valores EXPANDE (4.66 bits/peso).
   Los pesos cuantizados no tienen redundancia fila-a-fila que explotar.

**Directiva registrada:** validar POR MODELO antes de generalizar cualquier
receta de fracciones — en checkpoints no-topiary los canales no vienen
ordenados por saliencia (haría falta calcular orders y permutar el artefacto).

## 4. Piezas nuevas del runtime (este commit)

- `--centroid empirical`: suelo con centroide empírico por grupo (medido: no
  renta; documentado como negativo).
- `--p1-frac g,u,d`: fracción saliente del P1 en el pool por proyección,
  incluido 0 (proyección a puro suelo). Control verificado: 1.0 == comportamiento
  previo; tests 10/10.
- Fallback de checkpoint normal en `eval_stream` + shim per-layer (taper) en
  serve/eval vía PYTHONPATH del lab (`dense_loader`).
- Artefacto nuevo: `artifacts/qwen30-stream` (resident-p0 del campeón taper,
  8.4 GB residente + pools).
- Stage `bench` (math500/mmlu/mbpp/lambada) con verificación simbólica
  (math-verify), datasets congelados en `data/`.

## 5. El duelo contra Unsloth UD-Q2_K_XL (2026-08-22)

Misma máquina (M5 Pro 24 GB), misma batería, mismos prompts y parsers (vía
API de llama-server). El rival solo arranca con todo en CPU (`-ngl 0`, mmap):
con `-ngl 99` muere por OOM de Metal y con `--cpu-moe` el servidor muere
tras cargar. Fichero: 30.1 GB (28.05 GiB) — no cabe en 24 GB, pagina.

| | 80B Topiary Stream (C=240) | Unsloth UD-Q2_K_XL + llama.cpp |
|---|---|---|
| RAM servida | **17 GB** | 30 GB de pesos (pagina desde disco) |
| tok/s | **17.3–21.5** (GPU) | 12.6 ± 3.7 (CPU, tg128) |
| MATH-500 (n=100) | 64% | **69%** |
| MBPP (n=100) | 81% | **86%** |
| MMLU (n=500) | 85.2% | 86.2% (+5 ítems, n.s.) |
| Prompt | exacto (bit a bit) | 2-bit |

**Veredicto.** En calidad de tarea el estático calibrado está a la par o
nominalmente por encima (+5/+5/+1; n=100 en los generativos). Mi predicción
previa ("pierde 6–10 puntos") queda falsada: un 2-bit dinámico por capa
(~3 bpw efectivos, capas sensibles protegidas) mantiene la calidad de tarea.
Las ventajas de Stream son de SISTEMA: cabe donde el estático no cabe
(17 vs 30 GB), +40–70% de velocidad, prompt exacto, gobernador elástico y
suelo garantizado. Tesis corregida: la asignación temporal de bits es la
forma de correr bien un modelo de 42 GB en 24 GB — no una forma de superar
en calidad a un estático bien calibrado que necesita 30.

## 6. Métricas hostiles (KLD/trayectorias, 2026-08-22)

- KLD teacher-forced (T>1): **0.000** en 2278 tokens — identidad bit a bit
  del prefill exacto.
- Trayectorias greedy 300@32 (80B C=240 vs base exacta): exact-match 36%,
  divergencia media en el token 19.9 (Unsloth: su UD-Q2_K_XL ≈25% vs BF16).
- **KLD en decode (token a token, caché KV)**: LAMBADA (80 tok, prefijo 16):
  media 0.118, p99 1.83. **Wiki 4×512 (prefijo exacto 64): media 0.774,
  p95 4.40, p99 9.02** — el "+120% wiki" del peaje TF reaparece en decode:
  en dominio disperso los DROPS (fuera de C no hay suelo) destruyen. Suelo
  universal 2D del 80B (25% anchura, 6 GB, captura 61.6% de saliencia) a
  C=120: **1.354** — peor (suelo bajo el precipicio, como en el 235B); a
  C=240 no cabe (thrashing). Conclusión: a 24 GB el 80B no tiene suelo
  universal viable → sirve prompts y tareas a nivel base; NO prosa general
  larga. Pendientes: control drops C=120, modo 2-bit C=340, curva por
  posición, KLD del 30B-Stream (suelo universal real).
