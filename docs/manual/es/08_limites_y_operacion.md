# 8. Límites declarados y lecciones operativas

## 8.1 Límites del método

- **Formato**: solo cuantización afín plana (MLX). No porta a K-quants ni
  IQ-quants de GGUF; hacia arriba (Q8_0 → 2×Q4_0) sí.
- **Generalidad**: una máquina, una familia (Qwen: 3, 3.5, Next). El routing
  de decode de Qwen es muy local; un router balanceado por carga (Mixtral,
  OLMoE) podría derrotar a la residencia por recencia y debe medirse antes
  de reclamar las leyes de cobertura más allá de "MoE con alta localidad en
  decode". Por directiva el programa es solo-Qwen; la generalidad queda como
  limitación, no como tarea.
- **Fidelidad de prosa**: el 80B a 24 GB no es competitivo en KLD de texto
  general (0.774 vs 0.195 del estático); lo es en tareas y en sistema.
- **Cobertura**: todo slot enrutado necesita al menos un suelo; los drops
  desnudos de slots dominantes colapsan el modelo, no lo degradan. El 80B no
  tiene suelo universal viable a 24 GB.
- **Decodificación especulativa**: bloqueada en la generación híbrida de Qwen
  en mlx-lm actual (cachés SSM no recortables). Límite de implementación,
  no fundamental (*The Mamba in the Llama*).
- **Potencia estadística**: tareas a n=15–100; solo las igualdades de PPL y
  las KLD (miles de tokens) son fuertes. Ver §5.5.
- **Superficie de ingeniería**: un helper privado de mlx-lm
  (`_gather_sort`), semántica de `sorted_indices`, constructores parcheados;
  versiones fijadas.
- **Velocidad citable**: el 80B solo tiene cifra caliente (las rondas en frío
  disparan swap).
- **Almacenamiento frío**: paginar el suelo desde un disco USB fue >4× más
  lento (run abandonado). Los tiers lentos son para lecturas en refresh,
  nunca en el camino caliente.

- **Baseline del duelo.** Solo-CPU es la única forma de correr el GGUF de
  30 GB en 24 GB, pero el rival más fuerte nativo de Apple (flash-moe /
  Anemll, expertos streameados desde SSD, Metal propio, sin suelo en el
  miss) aún no se ha medido. Un 15/15 a n=15 solo acota el fallo por debajo
  del ~20 %.

## 8.2 Reglas duras de memoria

1. Comprobar RAM libre antes de cargar; abortar con mensaje claro si no hay
   presupuesto.
2. Loggear `mx.get_active_memory()` / `mx.get_peak_memory()` tras cada fase.
3. **Swap = 0 siempre**; si un benchmark provoca swapouts, el resultado se
   descarta y el run se marca inválido. Incidentes reales: duelo lanzado
   encima de f48 (11 GB de swap); OVF medido con 3–6 GB de swap (5.7 tok/s
   contaminado → re-medida limpia 5.8).
4. Los pools del 80B caben hasta C=290 (19.8 GB); C=340 (20.7 GB) no.
   floor2d a C=240 (pool 16.5 + suelo 6) hace thrashing.
5. El rival (30 GB paginados) y nuestro 80B (17 GB) nunca a la vez.

## 8.3 Cadenas desatendidas (`examples/*.sh`)

Patrón: `nohup` + `caffeinate` (+ Amphetamine), gate por **marcador** en
`runs/ablation.log` (`until grep -q "X COMPLETO" … && ! pgrep -f
"eval_stream|llama-|src/serve.py"; do sleep 60; done`), humo de un chunk
antes de cualquier run de horas, marcador `OK`/`FALLO` por etapa.

Lecciones pagadas:

- **Marcadores viejos abren puertas** (tres incidentes: `F48 FALLO` antiguo
  lanzó el duelo encima de f48; `RIVAL-KLD COMPLETO` del día anterior
  adelantó al OVF; un waiter viejo de `objetivo_kld` arrancó llama-server
  durante el OVF). Remedio: gates con recuento de marcadores (≥2) o
  marcadores con fecha; matar waiters obsoletos al reconfigurar la cola.
- **Un reinicio mata todas las cadenas** sin dejar rastro; detectarlo por
  `boottime` (`sysctl kern.boottime`, parseo con awk) y reorquestar
  (`morning_orchestrator.sh`).
- **Humo antes de horas**: un `KeyError` o una variable pisada (`out`
  sombreando la salida) tiraron runs de 46 min. Primer chunk + comprobar
  que el `.npz` existe.
- **llama-server**: `/health` ok no garantiza probabilidades; tokens UTF-8
  parciales en el borde del prompt devuelven respuestas sin
  `completion_probabilities` o HTTP 500 ("does not match the expected
  Content-only format"). El parser salta y cuenta.
- **Contar bien los tiempos**: con `cache_prompt` el rival evalúa un token
  por paso (1976 posiciones en 2.5 min); sin él, horas.

## 8.4 Descargas grandes (HF)

- `hf download` en **serie**, nunca en paralelo (se estrangulan); reiniciar
  una descarga viva abandona los parciales xet (74 GB huérfanos purgados);
  `du` engaña durante la descarga — medir con `netstat`/`lsof`.
- Un ceiling de hotspot de ~1 MB/s hace inútil bajar 132 GB; con Wi-Fi buena
  ~14 MB/s.
- El 235B se reconstruyó entero una vez por consumir los shards antes de
  escribir el esqueleto: ahora el esqueleto va primero.

## 8.5 Disco

- Las decisiones de borrado son del usuario. Cuando un `rm -rf` del
  artefacto del 35B fue bloqueado, se movió al disco externo y se verificó
  el sha256 contra HF.
- El externo se desconecta a veces: nada en el camino caliente puede vivir
  allí.
- Inventario y libres en §4.7.

## 8.6 Qué no hacer (resumen para el próximo que toque esto)

- No reconstruir el pool completo del 80B en cada refresh (10×).
- No dejar `inds` lazy en prefill bajo presión de memoria (índices basura).
- No evaluar con el pool congelado durante la generación (`--gen-refresh`).
- No comparar velocidades entre sesiones distintas (el "87 vs 104" fue un
  artefacto); rondas intercaladas, un proceso por medición.
- No citar cifras con swap.
- No generalizar recetas de subsampling entre modelos sin medirlas.
