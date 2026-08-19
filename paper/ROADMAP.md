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
   objetivo: igualar calidad sin duplicación ni dos rutas; (b) llama.cpp
   con mmap/offload para 35B/80B — cuantificar el "20 tok/s no tiene igual".

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
