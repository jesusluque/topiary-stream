#!/bin/zsh
# CONTROL: nosync (drops) a C=120 en el mismo KLD-long, para separar el
# efecto "pool a la mitad" del efecto "suelo de 25% de anchura".
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=/Users/muriel/luc/nanite-moe/data/calib_general_qwen3/held_out.jsonl
until grep -q "KLD-30B COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|f48_persistence" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== KLD-C120-DROP START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode nosync --pool-c 120 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80drop120_long >> runs/kld80.log 2>&1 && echo '=== KLD-C120-DROP OK ===' >> $LOG
echo "=== KLD-C120-DROP COMPLETO $(date) ===" >> $LOG
