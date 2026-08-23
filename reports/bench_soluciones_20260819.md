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

### 6b. La prueba del invariante: 30B-Stream (suelo universal)

Mismo test (wiki 4×512, prefijo exacto 64, token a token) sobre el 30B-Stream
(K=32 con P1; **P0 residente para los 128 expertos**) contra su propio taper
nativo: KLD media **0.131**, p95 0.52, p99 2.26 — **6× menor que el 80B con
drops (0.774)** a cobertura comparable de detalle. El suelo universal
convierte la catástrofe en un peaje moderado (los expertos fuera de K se
sirven a 2-bit: el precio del suelo, no del cero).

**Curva por posición** (media por tramo tras el prefijo): 0.228 (0–64) →
0.122 (64–128) → 0.132 (128–256) → 0.112 (256–448). El daño **no se
acumula con la longitud**: es un efecto de arranque (el pool aún no ha
aprendido el tema) que se estabiliza en ~0.12 tras los primeros ~64 tokens.
Remedio barato: refresh temprano tras el prompt; remedio de calidad: K mayor
para el régimen estacionario. Esto retira la hipótesis de "envenenamiento de
la caché KV" para el 30B; queda por ver la curva del 80B (en cola).

### 6c. Escalera de cobertura del 80B con la métrica hostil (wiki 4×512)

| Config 80B | Cobertura | KLD media | p99 | curva (0–64 → 256–448) |
|---|---|---|---|---|
| C=120 drops | 23% | 1.563 | 10.7 | 2.09 → 1.26 (decrece) |
| C=120 + suelo 25% | 23% (+suelo fino) | 1.354 | 9.5 | — |
| C=240 + K=32 (producción) | 47% | 0.774 | 9.0 | 0.88 → 0.94 → 0.79 → 0.68 (decrece) |
| C=290 todo P0 ("modo 2-bit"; C=340 no cabe: 20.7 GB) | 57% | **0.582** | 7.5 | 0.63 → 0.51 (decrece) |
| 30B-Stream, suelo universal | 100% P0 | **0.131** | 2.3 | 0.23 → 0.11 |

Lecturas: (1) la cobertura manda — la mitad de pool dobla el daño; (2) el
suelo fino (25%) ayuda un 13% a igual C, no es "peor que el drop" (mi lectura
de esta mañana comparaba C distintos); (3) las curvas DECRECEN con la
posición: el pool aprende; no hay acumulación en la KV — el daño es de
arranque y de régimen estacionario, no creciente.

### 6d. Palancas sobre el 80B C=240 (misma KLD realista)

| Política | KLD media | curva 0–64 → 256–448 | Veredicto |
|---|---|---|---|
| producción (nosync, refresh 256) | 0.774 | 0.88 → 0.68 | referencia |
| **refresh cada 64** | **0.416** | 0.62 → 0.38 | **−46%**: la cadencia es la palanca; coste medido −28% tok/s en el 80B |
| absorb (el shared absorbe la masa caída) | 7.17 | plano ~7 | **negativo rotundo**: sobrepondera al shared (escalas distintas) |
| modo 2-bit C=290 todo P0 | 0.582 | 0.63 → 0.51 | −25%: segunda marcha |

Hallazgo del arnés: las baterías de tareas corrían sin refresh
intra-generación (pool congelado ~500 tokens por respuesta). Corregido
(`--gen-refresh`); la campaña repite la batería con la mejor cadencia.

### 7. Campaña "superar la tabla" — primera vuelta (2026-08-22)

Cadencia de refresh (KLD wiki, C=240): 256→0.774 · 128→0.566 · 64→0.416 ·
32→0.303 (estacionario 0.18, bajo el listón 0.22). Velocidad: 128→15.4 ·
64→12.3 · 32→9.1 tok/s. Elegido refresh 128 (único ≥12.6). Batería con
refresh 128 INTRA-generación: **MATH 65% (antes 64), MBPP 81% (= antes)** —
la KLD mejora un 27% pero las tareas no se mueven: en dominio focal el pool
ya seguía al routing y el peaje de tareas no es de cadencia. El rival sigue
69/86. Hipótesis abierta: el peaje de tareas viene de la cobertura (drops
al 47%), no del refresh → probar C=290 (modo 2-bit) en tareas y el refresh
barato (tier de desbordamiento) que permita cadencia 32 a ≥15 tok/s.

### 7b. Refresh barato v1 — tier de desbordamiento: NEGATIVO (2026-08-23)

Idea: los expertos entrantes van a 32 filas pequeñas (copias ~27 MB) y el pool
grande solo se refresca cada 8 refreshes rápidos. Medido a cadencia 32
(wiki, C=240): **KLD 0.517** (vs 0.303 del refresh 32 plano y 0.566 del 128)
y **5.8 tok/s** limpios (vs 8.5 plano a 32, 17.1 a 256). Dos lecciones:
(1) el beneficio de la cadencia viene de refrescar el pool GRANDE (membresía
y detalle P1), no de tener al entrante a suelo en un tier aparte; (2) el
tier añade un gather_qmm por proyección y token, y 7 refreshes rápidos
(144 setitem+eval) por cada completo no salen gratis. Retirado de producción
(flag `--ovf-merge` queda a 0). Siguiente intento: refresh SELECTIVO por
capa (`--refresh-min-miss`): cadencia para todas, copias solo donde hay misses.

### 7c. Modo 2-bit (C=290, K=1) en tareas: NEGATIVO (2026-08-23)

MATH-500 n=100 con refresh 128 intra-gen: **52%** (C=240+K=32: 65%; rival
UD-Q2: 69%). La cobertura extra (57% vs 47%) a costa del detalle P1 **no
recupera tareas focales** — al contrario, −13. Conclusión: el peaje de
tareas del 80B frente al estático no es de cadencia (vuelta 1) ni de
cobertura-a-2-bits (vuelta 2). El modo 2-bit queda como marcha para texto
general (KLD 0.774→0.582), no para razonamiento. Lo que queda en pie como
explicación: la calidad del 2-bit que sirven los misses (P0 uniforme, el
peor suelo) — la dirección del máster protegido por saliencia (AWQ),
aparcada — o aceptar la frontera: a 24 GB el 80B empata en conocimiento y
gana en sistema, pero cede ~5 puntos en generativos.

Cierre C=290 (12:24): **MATH 52% · MBPP 70%** (pico 19.8 GB). Frente a
C=240+K32: −13 / −11. Veredicto definitivo: negativo en generativos.

### 6e. KLD del rival contra NUESTRA base exacta (2026-08-23, 12:50)

Unsloth UD-Q2_K_XL vía llama-server (CPU, `-ngl 0`), teacher-forced token a
token sobre los mismos wiki 4×512 y la misma referencia (base exacta del
80B guardada), KL truncada al top-100 de la referencia (aprox. estándar; 4
posiciones saltadas por UTF-8 parcial en el borde del prompt):

| | KLD media | p95 | p99 | tok/s | GB |
|---|---|---|---|---|---|
| UD-Q2_K_XL (estático, 30 GB, CPU) | **0.195** (0.189 en pos ≥64) | 0.91 | 1.94 | 12.6 | 30 |
| 80B Stream C=240, refresh 256 (producción) | 0.774 | 4.40 | 9.02 | 17–21 | 17 |
| 80B Stream, refresh 128 | 0.566 | 3.11 | 7.56 | 15.4 | 17 |
| 80B Stream, refresh 32 | 0.303 | 1.37 | 4.29 | 9.1 | 17 |
| 30B-Stream, suelo universal (otro modelo) | 0.131 | 0.52 | 2.26 | — | 11.8 |

Curva del rival por tramos (0–64 → 256–448): 0.270 → 0.221 → 0.152 → 0.176
(plana: no tiene "arranque" porque no tiene pool que aprender).

**Veredicto honesto:** en KLD de prosa el estático calibrado gana con
claridad — 4× a nuestra cadencia de producción y 1.6× a nuestra mejor
cadencia (que cuesta la mitad de la velocidad). Las colas (p99 1.9 vs 9.0)
son la diferencia real: nuestros DROPS fuera del pool producen tokens
catastróficos; su 2-bit calibrado nunca. Solo el suelo universal (30B-Stream,
0.131) baja de su cifra, y es otro modelo. Esto cierra la columna: el 80B a
24 GB es competitivo en tareas y superior en sistema, pero NO en fidelidad de
texto general. Nota de método: la KL truncada al top-100 subestima frente a
la KL completa de nuestras cifras; la diferencia no cambia el orden.
