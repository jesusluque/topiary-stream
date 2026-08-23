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
3c. **Duelo contra flash-moe / Anemll — BLOQUEADO same-model (23/08):** ambos
   motores solo soportan Qwen3.5-397B-A17B (arquitectura cableada); no hay
   camino para el 80B ni el 35B; un cruce 397B-en-su-motor vs 80B-en-Stream
   mediría plataformas, no métodos. Documentado en el paper §4 con sus
   cifras publicadas (4.4–12.9 tok/s; I/O de expertos bloqueante = 47 % del
   tiempo por token; sin PPL/KLD). Opción abierta (decisión usuario): correr
   su 397B en nuestro M5 de 24 GB como dato de plataforma (~150 GB de disco
   externo, horas). Texto original del punto: (expertos streameados desde SSD en
   Apple Silicon; Metal propio; sin suelo en el miss): el rival nativo de
   plataforma más fuerte, señalado por el análisis externo del 23/08. Es
   MEDIDA, no modificación: mismo M5, mismos prompts. Requiere disco externo
   (GGUF Q3 del 80B) y comprobar que soporte Qwen3-Next. Si no le ganamos
   claramente en calidad-por-GB a throughput comparable, el argumento de
   plataforma se debilita y el diferenciador queda en suelo + contabilidad.
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

## Estado 2026-08-23 (tras el duelo y la campaña de KLD) — CAMPAÑA CERRADA

**Cerrado esta semana (del roadmap):** 3a baseline Unsloth UD-Q2 (duelo
completo; corrección 23/08: el GGUF del 80B es de linaje Dynamic 2.0 según
su card, el texto 3.0 es solo referencia metodológica); citas añadidas tras
el análisis externo: MorphServe, flash-moe/Anemll, ELDR, ReMoE (todas
verificadas); 3b KLD/trayectorias (TF 0.000, decode 0.77→0.30 por cadencia,
traj 36%@32, curvas por posición); 4 cold-boot (35B); A dial C como perfil
de tarea (n=500); C medida de persistencia del 235B (78% a W=8).

**Objetivo "superar la tabla del rival" (69/86/86.2): CANCELADO por
decisión del usuario (12:08).** Regla: si no se alcanza al rival y cada
modificación no da rédito claro o empeora otros ejes, no se sigue. Posición
final: 65/81/85.2 @ 15.4 tok/s, 17 GB (rival 12.6 tok/s, 30 GB, CPU).

**Lo que la campaña dejó medido (todo en el paper §2.7):**
- Cadencia del refresh = LA palanca de KLD (256→32: 0.77→0.30) pero NO
  mueve tareas focales (65/81 a r128). Coste −10/−28/−47% tok/s.
- Cobertura a 2 bits (C=290, todo P0): KLD −25% en prosa; en tareas
  **MATH 52%, MBPP 70%** (−13/−9): NEGATIVO para razonamiento. Marcha de
  texto general solamente.
- Absorb (7.17), tier de desbordamiento (0.517 @ 5.8 tok/s), suelo fino
  25% (1.354 a C=120; no cabe a C=240): NEGATIVOS.
- Router ya va a 8 bits en mlx-community (corrección). `protect.py` (router
  BF16 / esqueleto 8-bit) construido pero NO medido — la regla de
  memoria/tiempo lo hacía un dial, no una mejora.
- Ráfaga, marchas, EMA por masa, prewarm, sensor de margen, refresh
  selectivo: **implementados como flags, NO medidos** (matados de la cola).
  No reclamar nada de ellos.

**Residuo explicativo del peaje generativo (~5 puntos):** ni cadencia ni
cobertura → la calidad del propio plano 2-bit servido en los misses (P0 de
ancla uniforme). Única palanca no probada: máster protegido por saliencia
(D, AWQ/imatrix afín) — aparcada por disco y por la regla anterior.

**J cerrado (12:50):** KLD del rival vs nuestra base exacta = **0.195**
(p99 1.94) frente a nuestro 0.774 (producción) / 0.303 (cadencia 32). En
fidelidad de prosa el estático gana 4×/1.6×; las colas (drops) son la
diferencia. Columna del paper cerrada (§2.7). Nada queda en marcha.

**Siguientes:** consolidación escrita (paper §2.7 hecho; README/web/cards)
→ n≥300 solo si se retoma la comparación → B/D cuando el usuario lo decida.

## Importado de Unsloth sin cambiar la filosofía (2026-08-23)

| Pieza de Unsloth | Versión nuestra | Estado |
|---|---|---|
| KLD + flips como métrica | stages kld/kld-decode/traj; **flips por ítem** (per_item + `flips()`, McNemar) | ✅ |
| Divergence-300@32 | stage traj (300 prompts math/mmlu/mbpp, greedy 32) | ✅ |
| Router nunca cuantizado | **Corrección:** mlx-community ya deja el router a 8 bits (override en config); `protect.py --router` lo sube a BF16 (8→16) → `qwen80-prot` — mejora esperada ≈0, test barato en cola | ✅ construido |
| Capas sensibles a más bits | **esqueleto no-experto a 8 bits** desde mlx-community-8bit por rangos (`--skeleton8`) → `qwen80-prot8`; overrides por ruta en `config.quantization` | 🟡 construyendo; test en cola |
| Calibración con chat-template, no solo texto | prior de residencia (orders_routed) recalculado con prompts chat-formateados | ⏳ pendiente (GPU) |
| imatrix / bits dentro del tensor | máster AWQ afín (`awq_master.py`, config qwen3_moe) | ⏸ aparcado por el usuario (disco) |
| Calibración ≠ evaluación (anti-overfitting) | calib_* para orders/EMA, held_out para KLD/PPL | ✅ ya era así |
| Quitar módulos muertos (MTP) | el esqueleto del 80B no lleva MTP (verificado) | n/a |
