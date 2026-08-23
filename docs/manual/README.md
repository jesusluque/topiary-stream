# Topiary Stream — Technical Manual

Spanish translation: [`es/`](es/README.md).

Detailed project manual: what it is, how it works inside, how the artifacts
are built and served, how everything is measured, what was measured, what
failed and how to reproduce it. It complements (does not replace) the root
`README.md` (public summary, English), the paper (`paper/topiary-stream.md`)
and the campaign reports (`reports/`).

Language: English is the reference version; identifiers, flags and file names appear exactly as in the code.

| Ch. | File | Contents |
|---|---|---|
| 1 | [01_vision_and_lineage.md](01_vision_and_lineage.md) | Mission, lineage (Topiary → Stream), corrected thesis, glossary |
| 2 | [02_method.md](02_method.md) | Anchored pyramid, artifacts, gate-governed pool, exact prefill, modes, governor |
| 3 | [03_runtime_internals.md](03_runtime_internals.md) | Code walkthrough: `common`, `fastpath`, `pager`, `serve` — state, patched forward, refresh, memory |
| 4 | [04_building_artifacts.md](04_building_artifacts.md) | `split`, `salience`, `floor`, `pyramid`, `protect`: commands, disk/time costs, artifact inventory |
| 5 | [05_evaluation_protocol.md](05_evaluation_protocol.md) | Determinism, `eval_stream` stages, frozen datasets, KLD/trajectories, duel, speed, statistics |
| 6 | [06_results.md](06_results.md) | Compendium of every measured table with its source file |
| 7 | [07_negatives.md](07_negatives.md) | Catalog of measured negatives and the lesson from each one |
| 8 | [08_limits_and_operations.md](08_limits_and_operations.md) | Declared limits, risks, operational lessons (memory, swap, downloads, overnight chains) |
| 9 | [09_reproduce.md](09_reproduce.md) | Claim-by-claim reproduction guide |
| 10 | [10_status_and_roadmap.md](10_status_and_roadmap.md) | Status as of 2026-08-23 and roadmap summary |

Honesty convention: **every figure has a file in
`runs/` and a command that regenerates it**; what was not measured is
declared as not measured; negatives are documented with the same rigor as
the positives.
