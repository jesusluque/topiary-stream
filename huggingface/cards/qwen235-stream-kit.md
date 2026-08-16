---
license: apache-2.0
base_model: Qwen/Qwen3-235B-A22B-Instruct-2507
library_name: mlx
tags:
- mlx
- moe
- topiary-stream
- quantized
- bit-planes
---

# qwen3-235b-topiary-stream-kit — build kit + an honest negative

This is NOT a servable model on 24 GB Apple Silicon, and this card says so
with measurements. It is the **build kit** for the day you (or we) have a
48–64 GB machine: skeleton checkpoint (4.5 GB, expert-free), universal 2D
salience floor (11.8 GB, k=256), routed-salience orders, and the plane
manifest. The 122 GB of expert plane memmaps are NOT uploaded — rebuild them
in ~1 h from the community checkpoint:

```bash
python src/split.py --src mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit-DWQ \
    --out artifacts/qwen235-stream --layout full-memmap --consume
# then drop this kit's files (floor, orders, README) into the artifact dir
```

## The measured verdict at 24 GB (four serving modes, six designs)

| Mode | tok/s | Output |
|---|---|---|
| pool + drop-renormalize | 12.0 | collapsed ("int int int…") |
| universal 16.7%-width salience floor | 5.7 | degenerate ("….") |
| blocking floor (full-width P0 on demand) | 0.2 | **perfect** |
| exact (batch/teacher-forced only) | ~9 eq. | true base quality |

Three converging walls, all measured: pool coverage (11–16% of experts ≪ the
~30–60-experts/layer working set), floor budget below the width cliff, and
flat per-expert salience (a 17% prefix captures only 53.5% of energy — vs
~94% on concentrated models). No pool size fits: C=20 already peaks at
22 GB. Token-level retry-on-miss does not converge (L·k·P(miss) ≫ 1 against
load-balanced expert tails; falsified live at 99% retries).

**Falsifiable prediction:** at 48–64 GB (pool coverage ≈50% + this kit's
floor as the safety tier) the same runtime serves this model with nothing new
built. If you own such a machine, this kit is a one-hour experiment.

## Provenance

Skeleton and floor built with `split.py` / `floor.py`; salience orders
profiled *through the exact-mode pager* (the source checkpoint had been
consumed — the memmaps are the model; 3,072 calibration tokens). Full story:
https://github.com/jesusluque/topiary-stream
