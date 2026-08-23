#!/bin/zsh
# Remedio del arranque: refresh cada 64 tokens (vs 256) en decode, C=240.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
until ! pgrep -f "eval_stream|llama-" > /dev/null; do sleep 30; done
echo "=== KLD-REFRESH64 START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --kld-refresh 64 --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80c240_r64 >> runs/kld80.log 2>&1 && echo '=== KLD-REFRESH64 OK ===' >> $LOG
echo "=== KLD-REFRESH64 COMPLETO $(date) ===" >> $LOG
