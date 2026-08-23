# 6. Compendio de resultados

Todas las cifras en M5 Pro 24 GB, MLX, greedy, semilla 1234. Cada tabla
indica el fichero de `runs/` o el informe del que sale. "n.s." = no
significativo al n indicado.

## 6.1 Los controles que lo anclan todo

| Control | Resultado | Fuente |
|---|---|---|
| P0+P1 == 4-bit a través del kernel | error 2.6e-7 | `tests/test_stream.py` |
| Q8 = ancla + refinamiento vs máster verdadero | PPL 3.9209 vs 3.9213 | lab, informe 09 |
| Q2 derivado por truncación vs Q2 nativo (OLMoE-7B) | ~20× peor PPL | lab, informe 09 |
| Pager τ=0 vs base 35B (TF-512) | 2.3614 vs 2.3623 (sin diferencia medible), pico 12.22 GB | README §Results |
| Modo exacto del 80B (base verdadera de un modelo que no cabe) | PPL 2.228 código / 5.557 general a 4.3 GB de pico | paper §2.2 |
| KLD teacher-forced con prefill exacto | 0.000 en 2278 tokens | `runs/kld_k80served.json` |
| Fold del centroide: media real del plano descartado | 1.504 (teórico 1.5) | paper §1.1 |

## 6.2 Las insignias: 35B y 80B

| | 35B (1.1× RAM) | 80B (1.8× RAM) |
|---|---|---|
| Tamaño 4-bit | 19.5 GB (Metal OOM al generar) | 42 GB |
| Pico servido | 12.5–14.6 GB | 16.5–17.9 GB |
| tok/s | 44.5 (K=64) · 47.6 (K=32) · 47.2 en frío citable | 17.3–21.5 caliente; 20.1 típico |
| PPL servida (prefill exacto) | 2.3614 / 7.0583 = base | 2.2280 / 5.5569 = base |
| HumanEval / GSM8K | 92 % / 92 % (n=25/50, pool-prefill) · 14/15 / 15/15 (exacto, n=15) | 15/15 / 15/15 (n=15; antes del prefill exacto 84 %/96 %) |
| MATH-500 / MBPP (n=100) | 60 / 78 | 64–65 / 81 |
| MMLU / LAMBADA (n=500) | 83.0 / 75.4 | 85.2 / 78.2 |

Fuentes: `runs/bench_b35_*.json`, `runs/bench_b80_*.json`,
`runs/bench_c80best_*.json`, `runs/tasks_*.json`.

## 6.3 Banco de soluciones: cinco modelos, cuatro ejes

| Bench | 30B original | 30B Topiary | 30B-Stream | 35B Stream | 80B Stream |
|---|---|---|---|---|---|
| RAM servida | 16.4–17.9 GB | 14.5–15.2 GB | **9.2–11.3 GB** | 12.5–14.6 GB | 16.5–17.9 GB |
| MATH-500 (100) | 70 | **72** | 67 | 60 | 64 |
| MBPP (100) | **83** | 81 | 82 | 78 | 81 |
| MMLU (500) | 78.2 | 68.2 | 69.4 | 83.0 | **85.2** |
| LAMBADA (500) | 64.6 | 60.2 | 60.0 | 75.4 | **78.2** |

Pares controlados: original/taper (mismo modelo): el taper conserva
razonamiento (n.s.) y cuesta **−10 MMLU** y −4.4 LAMBADA; +6 % tok/s y −2.7
GB. Taper/30B-Stream (mismo checkpoint): indistinguible en 3 de 4 ejes con
5.3 GB menos. Las demás columnas mezclan familias (direccionales).
Fuente: `runs/bench_b30*.json`, `reports/bench_soluciones_20260819.md` §1.

## 6.4 Velocidad (256 tokens, máquina caliente)

| Config | tok/s | Pico GB |
|---|---|---|
| 30B original nativo | 101.4–103.0 | 17.2 |
| 30B Topiary nativo | 108.1–108.3 | 14.5 |
| 30B-Stream K=32 | 47.9 | 11.1 |
| 35B K=4 / K=32 / K=64 | 54.8 / 47.6 / 44.5 | 12.9 / 13.9 / 15.0 |
| 35B refresh=64 | 46.4 | 13.9 |
| 35B + governor | 51.1 | 13.9 |
| 80B C=240 r256 / r128 / r64 / r32 | 17.1–17.3 / 15.4 / 12.3 / 8.5–9.1 | 17.0 |
| 80B C=120 | 28.1 | 10.2 |
| 80B C=290 K=1 | ~17 | 19.8 |
| 80B + tier desbordamiento r32 | 5.8 | — |
| Rival UD-Q2_K_XL (CPU) | 12.6 ± 3.7 | 30 GB paginados |

Atribución (35B): compilar el grafo no cambia nada (30.8 vs 30.4); servir solo
suelo (mitad de kernels y bytes) gana 3 %: el pool cuesta ~3 % y el resto es
el coste intrínseco batch-1 de la arquitectura (SSM híbrido + cabeza de 248k).

## 6.5 Régimen de decode: KLD contra la base exacta (wiki 4×512, prefijo 16)

| Config | Cobertura | KLD media | p95 | p99 | Curva 0–64 → 256–448 | Fuente |
|---|---|---|---|---|---|---|
| 80B C=120 drops | 23 % | 1.563 | — | 10.7 | 2.09 → 1.26 | `kld_k80drop120_long` |
| 80B C=120 + suelo 25 % | 23 % (+suelo) | 1.354 | — | 9.5 | — | `kld_k80floor120_long` |
| 80B C=240 K=32 r256 (producción) | 47 % | 0.774 | 4.40 | 9.02 | 0.88 → 0.94 → 0.79 → 0.68 | `kld_k80served_long` |
| 80B C=240 r128 | 47 % | 0.566 | 3.11 | 7.56 | — | `kld_cand_r128` |
| 80B C=240 r64 | 47 % | 0.416 | 2.01 | 5.55 | 0.62 → 0.38 | `kld_k80c240_r64` |
| 80B C=240 r32 | 47 % | 0.303 | 1.37 | 4.29 | estacionario 0.18 | `kld_cand_r32` |
| 80B C=290 todo P0 | 57 % | 0.582 | — | 7.5 | 0.63 → 0.51 | `kld_k80c340_long` |
| 80B absorb | 47 % | 7.17 | — | — | plano ~7 | `kld_k80absorb_long` |
| 80B tier desbordamiento r32 | 47 % | 0.517 | — | — | — | `kld_k80ovf_r32` |
| 30B-Stream K=32, P0 de todos | 100 % P0 | **0.131** | 0.52 | 2.26 | 0.23 → 0.12 → 0.13 → 0.11 | `kld_k30served_long` |
| **Rival UD-Q2_K_XL** (top-100, CPU) | estático | **0.195** | 0.91 | 1.94 | 0.27 → 0.22 → 0.15 → 0.18 | `kld_udq2_vs_base` |

Otras: LAMBADA en decode (80 tokens, prefijo 16) 80B: media 0.118, p99 1.83.
Trayectorias greedy 300@32 (80B C=240 vs base): exact-match 36 %, token
medio de divergencia 19.9 (Unsloth reporta ≈25 % para su UD-Q2_K_XL vs BF16).

Lecturas: la cobertura es el término de primer orden (la mitad de pool dobla
el daño); las curvas **decrecen** con la posición (daño de arranque, no
acumulación en la KV); la cadencia es el término de segundo orden y su
coste es velocidad; el suelo universal convierte la catástrofe en peaje
moderado.

## 6.6 El duelo contra Unsloth UD-Q2_K_XL (mismo 80B, misma máquina)

| | 80B Topiary Stream (C=240, r128) | Unsloth UD-Q2_K_XL + llama.cpp |
|---|---|---|
| RAM | **17 GB** | 30 GB (pagina desde disco) |
| tok/s | **15.4–21.5** (GPU) | 12.6 ± 3.7 (CPU) |
| MATH-500 (100) | 65 | **69** |
| MBPP (100) | 81 | **86** |
| MMLU (500) | 85.2 | 86.2 (+5 ítems, n.s.) |
| KLD prosa vs base (wiki) | 0.774 (r256) · 0.303 (r32) | **0.195** |
| Prompt | exacto | 2-bit |

Fuente: `runs/bench_udq2_*.json`, `runs/kld_udq2_vs_base.json`. Veredicto en
§1.3.

Fuentes: artefacto del rival https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF ; metodología y su listón
de KLD publicado (Gemma-27B vs BF16: Q2_K_XL 0.221, Q4_K_XL 0.024 — otro
modelo y otra referencia, no comparable directamente con nuestra columna)
https://unsloth.ai/docs/basics/dynamic-3.0-ggufs ; fundamento KLD/flips: *Accuracy is Not All You Need*
(arXiv:2407.09141).

## 6.7 Cobertura en tareas (80B, mismo checkpoint)

| Config | Cobertura | MATH-500 | MBPP | MMLU | LAMBADA | tok/s | Pico |
|---|---|---|---|---|---|---|---|
| C=120 | 23 % | 44 | 72 | 86.0 | 78.2 | 28.1 | 10.2–11.8 |
| C=240 K=32 (producción) | 47 % | 64–65 | 81 | 85.2 | 78.2 | 15.4–17.3 | 17.9 |
| C=290 K=1 (modo 2-bit) | 57 % | 52 | 70 | — | — | ~17 | 19.8 |

Lectura: la mitad de pool rompe el decode largo (−20 MATH, −9 MBPP) sin tocar
el conocimiento (MMLU/LAMBADA intactos); gastar el detalle en cobertura
(C=290) también rompe los generativos (−13/−11). Fuente: `bench_b80_c120_*`,
`bench_c80c290_hard`.

## 6.8 Subsampling saliente del P1 (30B-Stream, n=100)

| Config (g:u:d) | Bytes pool | MATH-500 | MBPP | MMLU | Pico |
|---|---|---|---|---|---|
| 4:4:4 (default) | 3.0× | 67 | 82 | 69.4 | 11.3 |
| 4:4:2 | 2.5× | 72 | — | — | 12.1 |
| 2:2:4 | 2.0× | 69 | 74 | 65.8 | 12.0 |
| 2:4:2 | 2.0× | 65 | — | — | 12.1 |
| 2:2:2 (K=64) | 3.0× | 66 | — | — | 13.8 |
| 2:2:2 | 1.5× | 65 | — | — | 12.0 |
| 1:1:4 | ~1.2× | 64 | — | — | 12.0 |
| 0:0:4 | ~1.0× | 64 | — | — | 10.4 |

Dial K en el 30B-Stream: K=4 65/75, MMLU 66.4 (10.6 GB); K=64 69/84, MMLU
69.2 (13.8 GB). Fuente: `runs/bench_b30s_*.json`.

## 6.9 El 235B (5.2× RAM): el negativo que mapea la frontera

| Modo | Resultado |
|---|---|
| drop-renormalize | 12 tok/s, salida colapsada |
| suelo universal 16.7 % (floor256) | texto degenerado a 5.7 tok/s (saliencia plana: 53.5 % capturado) |
| suelo bloqueante | texto perfecto a 0.2 tok/s (94 sincronizaciones/token) |
| exacto | solo batch |
| retry-on-miss por token | falsado el mismo día: 99 % de reintentos (condición real `L·k·P(miss) ≪ 1`) |
| persistencia del routing (f48) | W8 77.7 %, W16 86.8 % (el pool de seguimiento es viable en índices; sin suelo universal los ceros destruyen) |

Tres muros convergentes: cobertura 11–16 % ≪ working set de 30–60
expertos/capa; presupuesto de suelo bajo el precipicio de anchura; saliencia
plana. Predicción falsable: con 48–64 GB (cobertura ≈50 %) esta misma pila lo
sirve. Fuente: lab `runs/f44_*`, `f45_*`, `f47_*`, `f48_235b.log`.

## 6.10 Gobernador

Globo externo de 8 GB sobre el 35B: cuatro repliegues automáticos (K 32→4,
memoria activa 13.7→12.8 GB), generación completada a 49.3 tok/s. Fuente:
lab `runs/f46_balloon*.log`.
