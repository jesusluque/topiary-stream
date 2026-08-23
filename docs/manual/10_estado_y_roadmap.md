# 10. Estado a 2026-08-23 y roadmap

## 10.1 Estado

- **Runtime**: producción = 80B C=240 K=32 `nosync` refresh 128 (tareas) /
  256 (velocidad); 35B/30B resident-p0 K=32 con gobernador. Prefill exacto
  en todos. Tests verdes; CI fijado.
- **Evidencia cerrada**: controles de la pirámide; insignias 35B/80B; banco
  5×4; ley de cobertura en el mismo modelo (C=120/240/290); subsampling P1
  (8 puntos); 235B mapeado (4 modos + f48); métricas hostiles (KLD TF,
  decode, curvas, trayectorias); duelo completo contra Unsloth incluida su
  KLD contra nuestra base; gobernador demostrado; cold-boot del 35B.
- **Campaña "superar la tabla"**: cancelada el 23/08 por decisión del
  usuario tras dos vueltas sin rédito (cadencia no mueve tareas; cobertura a
  2 bits las empeora). Cuatro arreglos retirados; siete palancas
  implementadas sin medir (no se reclaman).
- **Escrito**: paper con §2.7 (régimen de decode y duelo), README, web,
  informe de campaña, roadmap, este manual. Cards HF de los repos privados
  sin la nota del duelo.
- **Cola de GPU**: vacía.

## 10.2 Roadmap (resumen; detalle en `paper/ROADMAP.md`)

**Pre-envío (medida pura, sin modificar el runtime):**

1. n≥300 por benchmark con IC bootstrap pareado — MATH-500 completo (500) y
   MBPP 257 en 80B y rival, en serie, ~12 h GPU + ~17 h CPU. Convierte "~5
   puntos n.s." en significativo o en indistinguible; no cambia direcciones.
2. Vendorizar `_gather_sort` (una hora).
3. Cold-boot del 80B sin swap (máquina recién reiniciada, nada más cargado).
4. Cards HF y flip a público cuando el usuario lo decida.

**Direcciones abiertas, por rédito esperado bajo la regla "sin rédito
claro, no":**

- **E. Perfil 16 GB**: empaquetar 30B-Stream (9.2 GB) / 0:0:4 (10.4 GB) con
  config conservadora y gobernador agresivo. Solo empaquetado y medida.
- **F. Modo exacto como servicio de evaluación**: PPL/KLD verdaderas de
  modelos insersibles con 4 GB de pico, en batch.
- **D. Máster protegido por saliencia (AWQ/imatrix afín)**: única palanca no
  probada para el residuo de calidad del plano 2-bit; necesita disco y horas
  de GPU.
- **B. Residencia guiada por daño**: retener a los expertos cuyo miss duele
  (las curvas KLD por posición ya existen para analizarlo antes de codificar).
- **C. Autoespeculación con el suelo como borrador**: solo MoE clásico
  (235B); bloqueada en híbridos.
- **G. Porte CUDA en AWS g6**: pirámide Q8 → 2×Q4 sobre kernels stock
  (Marlin/machete) en vLLM, residencia de tres tiers, réplica de la ley de
  cobertura; 1–2 semanas, $40–80. Contribuciones upstream: biases dinámicos
  a FusedMoE; split Q8_0→2×Q4_0 al RFC de llama.cpp.

Retirados: segunda familia (OLMoE) por directiva solo-Qwen; baseline HOBBIT
(omisión justificada en el paper); MLX-native AWQ aparcado.

## 10.3 Cómo seguir este manual al día

Cada resultado nuevo: fichero en `runs/`, fila en §6, y si es negativo
entrada en §7; si cambia un default, §2.7 y §10.1; si cambia un comando,
§9. El paper y el README se actualizan después, no antes.
