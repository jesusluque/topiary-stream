#!/bin/zsh
# Batería COMPLETA sobre la receta ganadora 2:2:4 (down entero, gate/up al 50%
# saliente): MBPP + MMLU + LAMBADA (MATH ya medido: 69%). K=32, 2/3 de bytes.
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
until grep -q "P1AXES COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 20 + 5))   # anti-carrera con el vigía del 30B original
pgrep -f "eval_stream" > /dev/null && exec "$0"
echo "=== 224 FULL START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench mbpp --n 100 --pool-k 32 --p1-frac 0.5,0.5,1.0 --tag b30s_224_mbpp >> runs/p1frac.log 2>&1 && echo '=== 224 MBPP OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench mmlu,lambada --n 500 --pool-k 32 --p1-frac 0.5,0.5,1.0 --tag b30s_224_wide >> runs/p1frac.log 2>&1 && echo '=== 224 WIDE OK ===' >> $LOG
echo "=== 224 FULL COMPLETO $(date) ===" >> $LOG
