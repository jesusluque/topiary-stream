#!/bin/zsh
# Router íntimo (2) y (3): precalentar el desbordamiento con casi-elegidos,
# y margen como sensor del gobernador. KLD wiki, C=240.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
REF=runs/kld80_base_long.npz
until grep -q "EMA-MASS COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== ROUTER-INTIMO START $(date) ===" >> $LOG
# (2) OVF cadencia 32 + precalentar 8 casi-elegidos (comparar con OVF r32 sin prewarm)
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --kld-refresh 32 --ovf-merge 8 --prewarm 8 --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref $REF --tag k80prewarm >> runs/kld80.log 2>&1 && echo '=== PREWARM OK ===' >> $LOG
# (3) gobernador con sensor de margen (vs sensor de misses del item 7)
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --kld-refresh 128 --gear 240:32,290:1 --gear-sensor margin --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref $REF --tag k80gear_margin >> runs/kld80.log 2>&1 && echo '=== GEAR-MARGIN OK ===' >> $LOG
echo "=== ROUTER-INTIMO COMPLETO $(date) ===" >> $LOG
