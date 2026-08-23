# Topiary Stream — Documentación técnica (versión en español)

> La versión de referencia es la inglesa: [`docs/manual/`](../README.md). Esta es su traducción.

Manual detallado del proyecto: qué es, cómo funciona por dentro, cómo se
construyen y sirven los artefactos, cómo se mide todo, qué se midió, qué
falló y cómo reproducirlo. Complementa (no sustituye) el `README.md` de la
raíz (resumen público, inglés), el paper (`paper/topiary-stream.md`) y los
informes de campaña (`reports/`).

Idioma: español (lengua de trabajo del proyecto); identificadores, flags y
nombres de ficheros en inglés tal como aparecen en el código.

| Cap. | Fichero | Contenido |
|---|---|---|
| 1 | [01_vision_y_lineaje.md](01_vision_y_lineaje.md) | Misión, lineaje (nanite-moe → Topiary → Stream), tesis corregida, glosario |
| 2 | [02_metodo.md](02_metodo.md) | Pirámide anclada, artefactos, pool gobernado por gate, prefill exacto, modos, gobernador |
| 3 | [03_runtime_internals.md](03_runtime_internals.md) | Recorrido por el código: `common`, `fastpath`, `pager`, `serve` — estado, forward parcheado, refresh, memoria |
| 4 | [04_construir_artefactos.md](04_construir_artefactos.md) | `split`, `salience`, `floor`, `pyramid`, `protect`: comandos, costes de disco/tiempo, inventario de artefactos |
| 5 | [05_protocolo_evaluacion.md](05_protocolo_evaluacion.md) | Determinismo, stages de `eval_stream`, datasets congelados, KLD/trayectorias, duelo, velocidad, estadística |
| 6 | [06_resultados.md](06_resultados.md) | Compendio de todas las tablas medidas con su fichero de origen |
| 7 | [07_negativos.md](07_negativos.md) | Catálogo de negativos medidos y la lección de cada uno |
| 8 | [08_limites_y_operacion.md](08_limites_y_operacion.md) | Límites declarados, riesgos, lecciones operativas (memoria, swap, descargas, cadenas nocturnas) |
| 9 | [09_reproducir.md](09_reproducir.md) | Guía de reproducción afirmación por afirmación |
| 10 | [10_estado_y_roadmap.md](10_estado_y_roadmap.md) | Estado a 2026-08-23 y resumen del roadmap |

Convención de honestidad, heredada del laboratorio: **cada cifra tiene un
fichero en `runs/` y un comando que la regenera**; lo no medido se declara
como no medido; los negativos se documentan con el mismo rigor que los
positivos.
