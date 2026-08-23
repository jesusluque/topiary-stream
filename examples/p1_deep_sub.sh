#!/bin/zsh
# Subsampling profundo sobre la ganadora: 1:1:4 (25% saliente) y 0:0:4
# (gate/up SIN detalle: puro suelo P0; down entero). MATH n=100.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
until grep -q "224 FULL COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 20 + 5)); pgrep -f "eval_stream" > /dev/null && exec "$0"
echo "=== DEEPSUB START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench math500 --n 100 --pool-k 32 --p1-frac 0.25,0.25,1.0 --tag b30s_114 >> runs/p1frac.log 2>&1 && echo '=== 1:1:4 OK (25% saliente) ===' >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench math500 --n 100 --pool-k 32 --p1-frac 0,0,1.0 --tag b30s_004 >> runs/p1frac.log 2>&1 && echo '=== 0:0:4 OK (gate/up puro suelo) ===' >> $LOG
echo "=== DEEPSUB COMPLETO $(date) ===" >> $LOG
