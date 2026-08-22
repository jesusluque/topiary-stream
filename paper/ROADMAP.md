# Pre-submission roadmap (tras la revisión externa de 2026-08-19)

Bloqueantes antes de enviar a venue (MLSys/EuroSys/ICML systems track):

1. (retirado por directiva: el programa es solo-Qwen; la generalidad entre
   familias queda declarada como limitación en el paper, no como tarea.)

2. **n≥300 por benchmark** con IC bootstrap (hoy: n=15/25/50/100; solo las
   igualdades de PPL son estadísticamente fuertes). McNemar solo entonces.
3. **Baselines directos en el mismo M5 Pro**: (a) mixed-precision estilo
   HOBBIT (copias hi/lo duplicadas + espera on-demand) vs nuestro P0 —
   objetivo: igualar calidad sin duplicación ni dos rutas; (b) **Unsloth UD-Q2 del 80B en llama.cpp, mismo M5** (sustituye al
   genérico "llama.cpp con mmap"): es la mejor versión del argumento
   contrario — un 2-bit estático imatrix que SÍ cabe en 24 GB. Tesis a
   probar: la asignación temporal de bits (camino caliente a 4-bit pleno)
   domina al downsizing estático a igualdad de memoria. Simetría honesta:
   su 2-bit machaca a nuestro P0 desnudo; la defensa es que P0 casi nunca
   se sirve.
3b. **KLD servido-vs-base + divergencia de trayectoria greedy** (estilo
   Divergence-300@32 de Unsloth, motivado por arXiv:2407.09141): nuestra
   evidencia fuerte es igualdad de PPL, exactamente la métrica que esa
   línea ataca. Tenemos el control de base exacta del 80B → medir KLD y
   flips es barato y blindaría el claim con la métrica del revisor hostil.

Fuertemente recomendado:

4. **Cold-boot**: tiempo a primer token servible tras reinicio + curva de
   warm-up del pool (el RFC de llama.cpp reporta ~22% de rampa; nosotros no
   lo hemos medido). Encaja con examples/citable_bench.sh (ya escrito,
   esperando reinicio).
5. **Vendorizar `_gather_sort`** (es pequeña) para inmunizar frente a
   mlx-lm 3.x. CI ya pinneado a mlx 0.32.0 / mlx-lm 0.31.3.
6. (retirado: iba unida a la segunda familia.)

Hecho el 2026-08-19 mismo (de la misma revisión): log-softmax estable en
token_nll, page-size parseado, pins de versión, nota del fold 1.5s
(óptimo-bajo-uniforme + medido casi-uniforme), limitations de potencia
estadística/generalidad/superficie de ingeniería, related work ampliado.

## Soluciones nuevas que la evidencia de esta semana abre (2026-08-20)

Ordenadas por cercanía; las dos primeras son tardes de trabajo sobre código
existente, no proyectos.

A. **Dial C como selector de perfil de tarea** (del hallazgo C=120: −20 MATH,
   ±0 MMLU, +62% tok/s, −40% RAM). El mismo artefacto sirve "modo
   conocimiento" (C=120) y "modo razonamiento" (C=240); un gobernador
   consciente del tipo de petición — o de la longitud de generación pedida —
   conmuta solo. Nadie tiene calidad-por-perfil como dial de runtime.
B. **Residencia guiada por daño (KLD-aware), no por popularidad**: si la KLD
   por token se concentra en misses concretos, el refresh puede retener a los
   expertos cuyo miss DUELE en vez de a los más pedidos. La telemetría ya
   existe; depende del resultado del stage kld.
C. **Autoespeculación con el suelo como borrador**: P0 es un draft model de
   memoria cero (mismos pesos, mitad de bytes, ya residente). Bloqueado en
   híbridos (caches SSM no recortables); viable en MoE clásico → el 235B:
   podría multiplicar el 0.2 tok/s del suelo bloqueante. Gate: la medición de
   persistencia f48 sobre el artefacto 235B.
D. **Híbrido Unsloth × Stream**: máster 4-bit con asignación imatrix (el suyo
   es afín plano → compatible con la partición en planos) + residencia
   encima. Ejes ortogonales que componen. Gate: el resultado del duelo UD-Q2.
E. **Perfil "small-machine" (gama 16 GB)**: 30B-Stream a 9.2 GB y 0:0:4 a
   10.4 GB de pico abren los Air/mini base — empaquetar artefacto + config
   conservadora + gobernador agresivo como producto de mayor audiencia.
F. **Modo exacto como servicio de evaluación**: PPL verdadera de un 42 GB con
   4.3 GB de pico → herramienta independiente para evaluar/destilar/generar
   datasets desde modelos insersibles, en batch.

G. **Porte CUDA en AWS (generalidad del método + upstream)**. La jerarquía
   VRAM/RAM/NVMe-efímero de las instancias GPU mapea 1:1 con
   pool/page-cache/memmaps — pero con DOS fronteras explícitas: el test duro
   de las leyes de cobertura (¿ley del routing o de la plataforma?).
   Plan: (1) pirámide subida un nivel — Q8 → 2 planos Q4 servibles por
   kernels STOCK (Marlin/machete int4; en CUDA no hay 2-bit de serie) sobre
   vLLM; (2) residencia de tres tiers con gobernador de dos fronteras;
   (3) réplica de la ley de cobertura + duelo vs llama.cpp/HOBBIT en la
   misma instancia. Instancias (generación g6, por decisión del usuario): g6.2xlarge
   (L4 24GB VRAM = espejo del presupuesto del M5, ~$1.0/h, Ada con mejor
   eficiencia que A10G) y g6e.xlarge (L40S 48GB, ~$1.9/h) que probaría de
   paso la predicción del 235B en la gama 48-64GB en silicio ajeno. Coste: ~30-60 GPU-h ≈ $40-80 +
   1-2 semanas. Contribuciones upstream independientes del porte: biases
   dinámicos como PR a vLLM/FusedMoE, y suelo Q8_0→2×Q4_0 (¡el split porta
   a GGUF por arriba, simétrico sin fold!) al RFC llama.cpp #24528.

## Estado 2026-08-23 (tras el duelo y la campaña de KLD)

**Cerrado esta semana (del roadmap):** 3a baseline Unsloth UD-Q2 (duelo
completo); 3b KLD/trayectorias (TF 0.000, decode 0.77→0.30 por cadencia,
traj 36%@32, curvas por posición); 4 cold-boot (35B); A dial C como perfil
de tarea (n=500); C medida de persistencia del 235B (78% a W=8).

**Objetivo activo — "superar la tabla del rival"** (69/86/86.2 en
MATH/MBPP/MMLU; 12.6 tok/s; 30 GB). Posición: 65/81/86 @ 15.4 tok/s, 17 GB.
Listón KLD (tabla Unsloth, Gemma-27B, wiki vs BF16): Q2_K_XL 0.221,
Q4_K_XL 0.024 → objetivo 80B ≤0.20 vs base con ≥12.6 tok/s (hoy 0.303 a
cadencia 32 pero 9.1 tok/s; 0.566 a 128 con 15.4).

**Hallazgos que reorientan:** (i) la cadencia del refresh es LA palanca de
KLD (256→32: 0.77→0.30) pero NO mueve tareas focales (65/81); (ii) el peaje
de tareas parece de COBERTURA (drops al 47%); (iii) el suelo universal fino
(25%) no rescata (1.35) y C=340 no cabe (20.7 GB); (iv) absorb (shared
como suelo) es negativo rotundo (7.17); (v) el 30B-Stream con suelo
universal da 0.131 — bajo el 2-bit: el invariante funciona.

**En cola / en marcha (examples/*.sh, automático):**
- H. **Refresh barato — tier de desbordamiento** (OVF=32 filas P0, merge
  cada N): ¿cadencia 32 a ≥15 tok/s? (ovf_test.sh). Si sí → producción.
- I. **Batería de tareas en modo 2-bit C=290** (57%): ¿la cobertura
  recupera los 5 puntos en MATH/MBPP? (bench_c290.sh).
- J. **KLD del rival contra nuestra base exacta** (kldremote, top-100):
  la misma columna para ambos (objetivo_kld.sh).
- K. Ráfaga de refresh tras el prompt (arranque ~64 tokens a 0.5–0.6 KLD):
  refresh cada 8–16 en los primeros 64 tokens.
- L. Gobernador de dos marchas (K↔C por tasa de misses) — validado el
  escalón (C=290: −25%), falta el conmutador.

**Siguientes (orden sugerido):** n≥300 en MATH/MBPP (el ±5 del duelo no se
cierra sin esto) → consolidación escrita (paper/README/web/cards, pendiente
desde el 20) → B residencia guiada por daño →
D máster imatrix afín (el P0 uniforme es el peor suelo posible).
