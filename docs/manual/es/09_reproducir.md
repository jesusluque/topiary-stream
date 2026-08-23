# 9. Reproducir, afirmación por afirmación

Prerrequisitos: M5 Pro 24 GB (o Apple Silicon ≥24 GB), macOS, `.venv` con
mlx 0.32.0 / mlx-lm 0.31.3 (`uv venv && uv pip install -e .`); el repositorio de
Topiary en `PYTHONPATH` solo para checkpoints Topiary; `llama.cpp` b10520 (brew)
para el rival. Todo desde la raíz de `topiary-stream`. Tiempos aproximados en
esa máquina.
desde la raíz de `topiary-stream`. Tiempos aproximados en esa máquina.

```bash
export PY=.venv/bin/python
export PYTHONPATH=<topiary>/src          # solo para checkpoints Topiary (anchos por capa)
export D=data/held_out_wiki.jsonl        # trozo held-out de WikiText-2, {"text": ...} por línea
```

## 9.1 "P0+P1 es bit-exacto y el suelo es el fold 1.5·s" (segundos)

```bash
$PY -m pytest -q tests/
```

## 9.2 "El 35B se sirve a 44–47 tok/s en 12–14 GB" (split 15 min; servicio 1 min)

```bash
$PY src/split.py --src mlx-community/Qwen3.5-35B-A3B-4bit --out artifacts/qwen35-stream --layout resident-p0
$PY src/serve.py --artifact artifacts/qwen35-stream --pool-k 32 --governor
# velocidad citable (tras reinicio limpio): examples/citable_bench.sh
```

## 9.3 "El 80B se sirve a 17–21 tok/s en 17 GB con PPL = base" (split ~1 h; PPL 20 min)

```bash
$PY src/split.py --src mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit --out artifacts/qwen80-stream --layout full-memmap --consume
$PY src/salience.py --artifact artifacts/qwen80-stream --data <calib.jsonl> --tokens 6000 --out artifacts/qwen80-stream/orders_routed.npz
$PY src/serve.py --artifact artifacts/qwen80-stream --pool-c 240 --pool-k 32 --orders artifacts/qwen80-stream/orders_routed.npz
# base verdadera sin cargar el modelo (4.3 GB de pico):
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage ppl --serve-mode exact --data-general $D --tag k80exact
# servida (prefill exacto): --serve-mode nosync --pool-c 240 --pool-k 32 --orders ...
```

## 9.4 Banco de soluciones (por configuración: hard ~2.5 h, wide ~1 h)

```bash
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage bench --bench math500,mbpp --n 100 \
    --serve-mode nosync --gen-refresh 128 --pool-c 240 --pool-k 32 --orders artifacts/qwen80-stream/orders_routed.npz --tag c80best_hard
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage bench --bench mmlu,lambada --n 500 \
    --serve-mode nosync --gen-refresh 128 --pool-c 240 --pool-k 32 --orders artifacts/qwen80-stream/orders_routed.npz --tag c80best_wide
# 30B-Stream / original / taper: examples/ablation.sh, examples/bench30orig.sh, examples/build30stream.sh
# cobertura C=120 / C=290: examples/speed_and_c120.sh, examples/bench_c290.sh
# subsampling P1: examples/p1frac_test.sh, p1frac_axes.sh, p1_224_full.sh, p1_deep_sub.sh, opt30_full.sh
```

## 9.5 KLD en régimen de decode (referencia 40 min; servida 45 min)

```bash
# referencia (base exacta, una vez):
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage kld --serve-mode exact --kld-decode \
    --data-general $D --chunks-general 4 --chunk-len 512 --kld-out runs/kld80_base_long.npz --tag k80base
# servida a cadencia N:
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage kld --serve-mode nosync --kld-decode --kld-refresh 256 \
    --pool-c 240 --pool-k 32 --orders artifacts/qwen80-stream/orders_routed.npz \
    --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80served_long
# variantes: --kld-refresh 128|64|32 ; --pool-c 290 --pool-k 1 ; --serve-mode absorb ; --ovf-merge 8
# curva por posición: runs/kld_<tag>_curve.npz ; scripts: examples/kld_*.sh
```

## 9.6 Trayectorias greedy 300@32 (~1 h)

```bash
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage traj --n 300 --gen-len 32 --serve-mode exact --kld-out runs/traj80_base.json --tag base
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage traj --n 300 --gen-len 32 --serve-mode nosync --pool-c 240 --pool-k 32 \
    --orders artifacts/qwen80-stream/orders_routed.npz --kld-ref runs/traj80_base.json --tag served
# examples/kld_traj_full.sh
```

## 9.7 El duelo (rival: ~8 h en CPU) y su KLD (3 min)

```bash
# descarga (en serie, Wi-Fi buena): hf download unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF --include "*UD-Q2_K_XL*" --local-dir artifacts/unsloth
./examples/duel_udq2_v3.sh          # llama-server -ngl 0; bench con --openai-base
./examples/objetivo_kld.sh          # kldremote contra runs/kld80_base_long.npz (gate por marcador)
```

## 9.8 Gobernador bajo presión (5 min)

```bash
$PY src/serve.py --artifact artifacts/qwen35-stream --pool-k 32 --governor --max-tokens 512 &
$PY examples/balloon.py             # globo de 8 GB en otro proceso; observar "[gov] avail … K a->b"
```

## 9.9 El 235B (descarga 132 GB; split horas; cada modo minutos)

```bash
$PY src/split.py --src mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit-DWQ --out artifacts/qwen235-stream --layout full-memmap --consume
$PY src/salience.py --artifact artifacts/qwen235-stream --data <calib> --out artifacts/qwen235-stream/orders_routed.npz
$PY src/floor.py --artifact artifacts/qwen235-stream --orders artifacts/qwen235-stream/orders_routed.npz --k-floor 256 --out artifacts/qwen235-stream/floor256.safetensors
$PY src/serve.py --artifact artifacts/qwen235-stream --pool-c 64 --pool-k 8 --serve-mode nosync|floor|floor2d --floor artifacts/qwen235-stream/floor256.safetensors
```

## 9.10 Ficheros de verdad

`runs/*.json` (métricas), `runs/*_curve.npz` (KLD por posición),
`runs/ablation.log` (marcadores con fecha de cada etapa), `runs/*.log`
(salida completa), `reports/bench_soluciones_20260819.md` (informe de
campaña con las lecturas), `paper/` (paper, bibliografía, roadmap). Las
configuraciones van en la línea de comandos de cada script de `examples/`
(congeladas en git).
