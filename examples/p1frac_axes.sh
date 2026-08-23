#!/bin/zsh
# ¿Qué proyección es la "luma"? Reparto asimétrico del detalle P1 (MATH n=100).
# Notación (gate:up:down). Baselines ya medidos: 4:4:4=67%, 2:2:2=66%(K64) y K32 en curso.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
until grep -q "P1FRAC COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py" > /dev/null; do sleep 60; done
echo "=== P1AXES START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench math500 --n 100 --pool-k 32 --p1-frac 0.5,0.5,1.0 --tag b30s_224 >> runs/p1frac.log 2>&1 && echo '=== 2:2:4 OK (down entero, gate/up a mitad) ===' >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench math500 --n 100 --pool-k 32 --p1-frac 1.0,1.0,0.5 --tag b30s_442 >> runs/p1frac.log 2>&1 && echo '=== 4:4:2 OK (down a mitad, gate/up enteros) ===' >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench math500 --n 100 --pool-k 32 --p1-frac 0.5,1.0,0.5 --tag b30s_242 >> runs/p1frac.log 2>&1 && echo '=== 2:4:2 OK (up entero) ===' >> $LOG
echo "=== P1AXES COMPLETO $(date) ===" >> $LOG
