# 7. Catálogo de negativos medidos

Cada entrada: la idea, la medida que la mató, y la lección. Los negativos
están documentados con el mismo rigor que los positivos porque acotan el
espacio de diseño tan bien como ellos.

## 7.1 Pirámide derivada por truncación (laboratorio)

- **Idea**: derivar Q4/Q2 de un máster Q8 por desplazamiento de bits (una
  sola copia, niveles anidados gratis).
- **Medida**: gana en L2 y error máximo de pesos; Q2 derivado ~20× peor PPL
  que un Q2 nativo (OLMoE-7B).
- **Lección**: el error del grid truncado cae sobre los pesos salientes; la
  pirámide se ancla en el nivel de servicio y se refina hacia arriba.

## 7.2 Centroide empírico por grupo (`--centroid empirical`)

- **Idea**: sustituir el fold uniforme 1.5·s por la media real del plano
  descartado por grupo (mismo formato, solo cambia la constante).
- **Medida**: mejora del MSE del suelo 1.5 %; media global 1.504.
- **Lección**: los 2 bits bajos son ruido casi uniforme; no hay información
  que recuperar ahí. El suelo mejora por saliencia (qué pesos), no por
  constantes (qué valores).

## 7.3 Compresión delta entre filas

- **Idea** (externa): máscara de 1 bit de cambios entre filas vecinas +
  valores cambiados, como un códec de vídeo.
- **Medida**: igualdad entre filas vecinas o expertos vecinos = tasa de azar
  exacta (8.6 % medido vs 8.6 % de la distribución de códigos q4); el
  esquema expande a 4.66 bits/peso.
- **Lección**: los pesos cuantizados entrenados no tienen redundancia
  fila-a-fila; la estructura explotable es *qué* pesos importan.

## 7.4 Subsampling saliente del P1 como optimización general

- **Idea**: servir P1 solo para el prefijo saliente (4:2:2, 2:2:4…).
- **Medida**: MATH casi plano de 3.0× a 1.0× bytes (67→64), pero la batería
  completa del 2:2:4 paga −8 MBPP y −3.6 MMLU.
- **Lección**: dial de emergencia de memoria (0:0:4 sirve MATH usable a 10.4
  GB), no default. Validar por modelo: en checkpoints no-Topiary los canales
  no vienen ordenados.

## 7.5 Prefill servido por el pool

- **Medida**: +6–11 % PPL en el 35B, +28 %/+120 % en el 80B; tareas del 80B a
  84 %/96 %.
- **Lección**: el routing de prefill es plano y derrota a la recencia;
  servir el prompt exacto (es una pasada batcheada) borra el peaje entero.
  Es el negativo que se convirtió en diseño.

## 7.6 Suelo universal fino

- **Medida**: 80B al 25 % de anchura (6 GB, captura 61.6 %): KLD 1.354 a
  C=120 (mejor que los drops un 13 %, pero bajo el precipicio); no cabe junto
  al pool de C=240 (thrashing, 0.1 tok/s). 235B al 16.7 % (53.5 %): texto
  degenerado.
- **Lección**: la anchura del suelo debe capturar ≥85–90 % de la saliencia
  (≈50 % de anchura en estos modelos) para ser un nivel de calidad; por
  debajo, coherencia de último recurso.

## 7.7 Retry-on-miss por token (235B)

- **Idea**: reintentar el token cuando falta un experto, esperando que
  converja con C ≫ working set.
- **Medida**: 99 % de reintentos.
- **Lección**: la condición es `L·k·P(miss) ≪ 1`, inalcanzable en modelos
  profundos con colas de expertos balanceadas por carga.

## 7.8 Absorb (el experto compartido como suelo)

- **Idea**: transferir la masa de gate de los expertos caídos al experto
  compartido, ya residente y entrenado.
- **Medida**: KLD 7.17 (plano ~7 a lo largo de la secuencia).
- **Lección**: las escalas de salida del compartido y de un experto enrutado
  no son comparables; sobrepondera al compartido y destruye.

## 7.9 Tier de desbordamiento (refresh barato, `--ovf-merge`)

- **Idea**: los entrantes van a 32 filas pequeñas (copias ~27 MB) a cadencia
  fina; el pool grande solo se refresca cada N rápidos.
- **Medida**: KLD 0.517 a cadencia 32 (refresh 32 plano: 0.303; 128: 0.566)
  y 5.8 tok/s limpios (8.5 plano a 32; 17.1 a 256).
- **Lección**: el beneficio de la cadencia viene de refrescar el pool
  **grande** (pertenencia y detalle P1); el tier añade un `gather_qmm` por
  proyección y token y 7 refreshes rápidos por cada completo no salen gratis.

## 7.10 Modo 2-bit (C=290, K=1) para tareas

- **Idea** (LOD dinámico): gastar los bytes del detalle en cobertura.
- **Medida**: KLD de prosa −25 % (0.774→0.582), pero MATH 52 (vs 65) y MBPP
  70 (vs 81).
- **Lección**: en régimen focal el detalle 4-bit de los calientes vale más
  que +50 expertos a 2 bits. Marcha de texto general, no de razonamiento.

## 7.11 Cadencia como remedio de tareas

- **Medida**: refresh 256→32 baja la KLD 0.774→0.303; refresh 128
  intra-generación deja MATH 65 / MBPP 81 (sin cambio); coste −10/−28/−47 %
  tok/s.
- **Lección**: en prompts focales el pool ya seguía al routing; la KLD
  mejora en tokens que las tareas no puntúan. La cadencia es palanca de
  fidelidad de prosa, con precio.

## 7.12 La hipótesis de la caché envenenada (decode)

- **Medida**: curvas por posición decrecientes (80B 0.88→0.68; 30B
  0.23→0.11).
- **Lección**: el daño es de arranque (el pool aún no conoce el tema), no se
  acumula en la KV. Retirada.

## 7.13 Predicción "el 2-bit estático pierde 6–10 puntos"

- **Medida**: Unsloth UD-Q2_K_XL 69/86/86.2 vs nuestro 65/81/85.2; KLD 0.195
  vs 0.774.
- **Lección**: un 2-bit dinámico por capa con imatrix (~3 bpw, capas
  sensibles protegidas) mantiene la calidad. La ventaja de Stream es de
  sistema (§1.3).

## 7.14 Lo que no se llegó a medir (y no se reclama)

Ráfaga tras el prompt, gobernador de dos marchas, EMA por masa,
precalentamiento con casi-elegidos, sensor de margen, refresh selectivo por
capa, router BF16 / esqueleto 8-bit. Implementados; cancelados por la regla
"sin rédito claro, o con bajadas en otros ejes, no se sigue".

## 7.15 El residuo

Tras descartar cadencia y cobertura, lo que explica el peaje generativo de
~5 puntos y las colas de KLD (p99 9.0 vs 1.9) es la calidad del propio plano
2-bit servido en los misses: P0 es un plano de ancla uniforme, el peor suelo
posible. La única palanca no probada es un máster protegido por saliencia
(AWQ/imatrix afín) — aparcada por disco y por decisión del usuario.
