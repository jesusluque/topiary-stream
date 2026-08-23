#!/bin/zsh
# TODAS las optimizaciones sobre el 30B-Stream: dial K (4/32/64), cadencia de
# refresh, governor (velocidad); y calidad COMPLETA (4 ejes) en los extremos
# del dial K. El K=32 4:4:4 ya tiene batería (67/82/69.4/60).
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen30-stream
until grep -q "C120 COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 20 + 5)); pgrep -f "eval_stream" > /dev/null && exec "$0"
echo "=== OPT30 START $(date) ===" >> $LOG

run() {
  echo "--- $1" >> $LOG
  local name=$1; shift
  $PY -u src/serve.py --artifact $A --max-tokens 256 ${=@} > /tmp/o30.txt 2>&1
  grep -E "tok/s" /tmp/o30.txt | tail -1 >> $LOG; rm -f /tmp/o30.txt
}
run "30s_K4"        "--pool-k 4  --refresh 256"
run "30s_K32_base"  "--pool-k 32 --refresh 256"
run "30s_K64"       "--pool-k 64 --refresh 256"
run "30s_refresh64" "--pool-k 32 --refresh 64"
run "30s_governor"  "--pool-k 32 --refresh 256 --governor"
echo "=== OPT30 SPEED OK ===" >> $LOG

$PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 100 --pool-k 4 --tag b30s_k4_hard >> runs/opt30.log 2>&1 && echo '=== K4 HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench mmlu,lambada --n 500 --pool-k 4 --tag b30s_k4_wide >> runs/opt30.log 2>&1 && echo '=== K4 WIDE OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 100 --pool-k 64 --tag b30s_k64_hard >> runs/opt30.log 2>&1 && echo '=== K64 HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench mmlu,lambada --n 500 --pool-k 64 --tag b30s_k64_wide >> runs/opt30.log 2>&1 && echo '=== K64 WIDE OK ===' >> $LOG
echo "=== OPT30 COMPLETO $(date) ===" >> $LOG
