# Pre-submission roadmap (tras la revisión externa de 2026-08-19)

Bloqueantes antes de enviar a venue (MLSys/EuroSys/ICML systems track):

1. **Segunda familia de modelos** con router balanceado (OLMoE-1B-7B es el
   candidato: ya usado como banco en el programa; alternativa Mixtral-8x7B).
   Umbral: si suelo + leyes de cobertura se sostienen → generalidad
   defendida; si no → acotar el claim a "MoE con localidad de decode alta"
   y demostrarlo con Gini/skew del routing.
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
6. **Ablación EMA bajo router balanceado** (tamaño de refresh N, K) para
   acotar thrashing.

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
