#!/bin/zsh
# Experimento 4:2:2: P1 recortado al prefijo saliente. MATH-500 n=100
# (el eje sensible al suelo) en dos puntos vs baseline 67% (K=32, P1 completo).
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
until ! pgrep -f "eval_stream|src/serve.py" > /dev/null; do sleep 30; done
echo "=== P1FRAC START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench math500 --n 100 --pool-k 64 --p1-frac 0.5 --tag b30s_k64p05 >> runs/p1frac.log 2>&1 && echo '=== K64 P1-0.5 OK (mismos bytes que K32 completo) ===' >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench math500 --n 100 --pool-k 32 --p1-frac 0.5 --tag b30s_k32p05 >> runs/p1frac.log 2>&1 && echo '=== K32 P1-0.5 OK (mitad de bytes) ===' >> $LOG
echo "=== P1FRAC COMPLETO $(date) ===" >> $LOG
