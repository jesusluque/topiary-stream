#!/bin/zsh
# Router íntimo (1): EMA por masa de gate. KLD wiki r128 C=240 vs 0.566.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
until grep -q "PROT COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== EMA-MASS START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --kld-refresh 128 --ema-mass --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80emamass >> runs/kld80.log 2>&1 && echo '=== EMA-MASS OK ===' >> $LOG
echo "=== EMA-MASS COMPLETO $(date) ===" >> $LOG
