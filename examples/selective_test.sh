#!/bin/zsh
# Refresh SELECTIVO por capa: cadencia 32 pero solo se refrescan (copias) las
# capas con misses >= 5%. Velocidad (256 tok) + KLD wiki. Gate: tras la KLD
# del rival re-encolada (segundo marcador COMPLETO).
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=/Users/muriel/luc/nanite-moe/data/calib_general_qwen3/held_out.jsonl
until [ $(grep -c "RIVAL-KLD COMPLETO" runs/ablation.log 2>/dev/null) -ge 2 ] && ! pgrep -f "eval_stream|llama-|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== SELECTIVE START $(date) swap=$(sysctl -n vm.swapusage | awk '{print $6}') ===" >> $LOG
for mm in 0.05 0.15; do
  $PY -u src/serve.py --artifact $A --pool-c 240 --pool-k 32 --orders $ORD --refresh 32 --refresh-min-miss $mm --max-tokens 256 2>&1 | grep "tok/s" | sed "s/^/[speed sel$mm r32] /" >> runs/campana.log
  $PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --kld-refresh 32 --refresh-min-miss $mm --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80sel$mm >> runs/kld80.log 2>&1 && echo "=== SELECTIVE $mm OK ===" >> $LOG
done
echo "=== SELECTIVE COMPLETO $(date) ===" >> $LOG
