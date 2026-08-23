#!/bin/zsh
# Hipótesis "el peaje de tareas es de COBERTURA": batería hard (MATH/MBPP
# n=100) en modo 2-bit C=290 (57%, K=1) con refresh 128 intra-gen.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
until grep -qE "OVF (COMPLETO|HUMO FALLO)" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== BENCH-C290 START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 100 --serve-mode nosync --gen-refresh 128 --pool-c 290 --pool-k 1 --orders $ORD --tag c80c290_hard >> runs/campana.log 2>&1 && echo '=== BENCH-C290 OK ===' >> $LOG
echo "=== BENCH-C290 COMPLETO $(date) ===" >> $LOG
