# 9. Reproduce, claim by claim

Prerequisites: M5 Pro 24 GB (or Apple Silicon ≥24 GB), macOS, `.venv` with
mlx 0.32.0 / mlx-lm 0.31.3, `PYTHONPATH=$LAB/src`
for Topiary checkpoints, `llama.cpp` b10520 (brew) for the rival. Everything
from the root of `topiary-stream`. Approximate times on that machine.

```bash
export LAB=<path to the nanite-moe lab>   # private; only needed for Topiary checkpoints and the calibration corpora
export PY=$LAB/.venv/bin/python
export PYTHONPATH=$LAB/src
export D=$LAB/data/calib_general_qwen3/held_out.jsonl
```

## 9.1 "P0+P1 is bit-exact and the floor is the 1.5·s fold" (seconds)

```bash
$PY -m pytest -q tests/
```

## 9.2 "The 35B serves at 44–47 tok/s in 12–14 GB" (split 15 min; serving 1 min)

```bash
$PY src/split.py --src mlx-community/Qwen3.5-35B-A3B-4bit --out artifacts/qwen35-stream --layout resident-p0
$PY src/serve.py --artifact artifacts/qwen35-stream --pool-k 32 --governor
# citable speed (after a clean reboot): examples/citable_bench.sh
```

## 9.3 "The 80B serves at 17–21 tok/s in 17 GB with PPL = base" (split ~1 h; PPL 20 min)

```bash
$PY src/split.py --src mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit --out artifacts/qwen80-stream --layout full-memmap --consume
$PY src/salience.py --artifact artifacts/qwen80-stream --data <calib.jsonl> --tokens 6000 --out artifacts/qwen80-stream/orders_routed.npz
$PY src/serve.py --artifact artifacts/qwen80-stream --pool-c 240 --pool-k 32 --orders artifacts/qwen80-stream/orders_routed.npz
# true base without loading the model (4.3 GB peak):
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage ppl --serve-mode exact --data-general $D --tag k80exact
# served (exact prefill): --serve-mode nosync --pool-c 240 --pool-k 32 --orders ...
```

## 9.4 The solutions bench (per configuration: hard ~2.5 h, wide ~1 h)

```bash
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage bench --bench math500,mbpp --n 100 \
    --serve-mode nosync --gen-refresh 128 --pool-c 240 --pool-k 32 --orders artifacts/qwen80-stream/orders_routed.npz --tag c80best_hard
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage bench --bench mmlu,lambada --n 500 \
    --serve-mode nosync --gen-refresh 128 --pool-c 240 --pool-k 32 --orders artifacts/qwen80-stream/orders_routed.npz --tag c80best_wide
# 30B-Stream / original / taper: examples/ablation.sh, examples/bench30orig.sh, examples/build30stream.sh
# coverage C=120 / C=290: examples/speed_and_c120.sh, examples/bench_c290.sh
# P1 subsampling: examples/p1frac_test.sh, p1frac_axes.sh, p1_224_full.sh, p1_deep_sub.sh, opt30_full.sh
```

## 9.5 KLD in the decode regime (reference 40 min; served 45 min)

```bash
# reference (exact base, once):
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage kld --serve-mode exact --kld-decode \
    --data-general $D --chunks-general 4 --chunk-len 512 --kld-out runs/kld80_base_long.npz --tag k80base
# served at cadence N:
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage kld --serve-mode nosync --kld-decode --kld-refresh 256 \
    --pool-c 240 --pool-k 32 --orders artifacts/qwen80-stream/orders_routed.npz \
    --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80served_long
# variants: --kld-refresh 128|64|32 ; --pool-c 290 --pool-k 1 ; --serve-mode absorb ; --ovf-merge 8
# per-position curve: runs/kld_<tag>_curve.npz ; scripts: examples/kld_*.sh
```

## 9.6 Greedy trajectories 300@32 (~1 h)

```bash
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage traj --n 300 --gen-len 32 --serve-mode exact --kld-out runs/traj80_base.json --tag base
$PY src/eval_stream.py --artifact artifacts/qwen80-stream --stage traj --n 300 --gen-len 32 --serve-mode nosync --pool-c 240 --pool-k 32 \
    --orders artifacts/qwen80-stream/orders_routed.npz --kld-ref runs/traj80_base.json --tag served
# examples/kld_traj_full.sh
```

## 9.7 The duel (rival: ~8 h on CPU) and its KLD (3 min)

```bash
# download (serially, good Wi-Fi): hf download unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF --include "*UD-Q2_K_XL*" --local-dir artifacts/unsloth
./examples/duel_udq2_v3.sh          # llama-server -ngl 0; bench with --openai-base
./examples/objetivo_kld.sh          # kldremote against runs/kld80_base_long.npz (gate by marker)
```

## 9.8 Governor under pressure (5 min)

```bash
$PY src/serve.py --artifact artifacts/qwen35-stream --pool-k 32 --governor --max-tokens 512 &
$PY examples/balloon.py             # 8 GB balloon in another process; watch "[gov] avail … K a->b"
```

## 9.9 The 235B (132 GB download; split hours; each mode minutes)

```bash
$PY src/split.py --src mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit-DWQ --out artifacts/qwen235-stream --layout full-memmap --consume
$PY src/salience.py --artifact artifacts/qwen235-stream --data <calib> --out artifacts/qwen235-stream/orders_routed.npz
$PY src/floor.py --artifact artifacts/qwen235-stream --orders artifacts/qwen235-stream/orders_routed.npz --k-floor 256 --out artifacts/qwen235-stream/floor256.safetensors
$PY src/serve.py --artifact artifacts/qwen235-stream --pool-c 64 --pool-k 8 --serve-mode nosync|floor|floor2d --floor artifacts/qwen235-stream/floor256.safetensors
# token-by-token persistence: nanite-moe/src/f48_persistence.py (lab)
```

## 9.10 Source-of-truth files

`runs/*.json` (metrics), `runs/*_curve.npz` (per-position KLD),
`runs/ablation.log` (dated markers for each stage), `runs/*.log`
(full output), `reports/bench_soluciones_20260819.md` (campaign report
with the readings), `paper/` (paper, bibliography, roadmap). The
configurations live on the command line of each script in `examples/`
(frozen in git).
