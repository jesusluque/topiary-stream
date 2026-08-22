#!/bin/zsh
# ¿El peaje de 0.77 nats era por los DROPS? Mismo test (wiki 4×512, prefijo
# 64) con el 80B en modo floor2d: suelo universal (P0 × prefijo saliente 25%)
# para TODOS los expertos + pool. Solo la pasada servida (la base existe).
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
F=artifacts/qwen80-floor128.safetensors
D=/Users/muriel/luc/nanite-moe/data/calib_general_qwen3/held_out.jsonl
until grep -q "\[out\]" runs/floor80.log 2>/dev/null && [ -f $F ] && ! pgrep -f "floor.py|eval_stream|llama-" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== KLD-FLOOR START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode floor2d --floor $F --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 1 --chunk-len 128 --kld-out runs/kld_smoke.npz --tag smoke >> runs/kld80.log 2>&1
[ -f runs/kld_smoke.npz ] || { echo '=== KLD-FLOOR HUMO FALLO ===' >> $LOG; exit 1; }; rm -f runs/kld_smoke.npz
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode floor2d --floor $F --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80floor_long >> runs/kld80.log 2>&1 && echo '=== KLD-FLOOR SERVIDO OK ===' >> $LOG
echo "=== KLD-FLOOR COMPLETO $(date) ===" >> $LOG
