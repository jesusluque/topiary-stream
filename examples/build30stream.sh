#!/bin/zsh
# Versión Stream del campeón Topiary: split resident-p0 + smoke + suite 4 ejes.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
M30=${LAB:-$HOME/luc/nanite-moe}/models/qwen3-30b-4bit-fine-taper085

until grep -q "SOLUCIONES v2 OK" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py" > /dev/null; do
  sleep 60
done
echo "=== 30B-STREAM BUILD START $(date) ===" >> $LOG
$PY -u src/split.py --src $M30 --out artifacts/qwen30-stream --layout resident-p0 >> runs/build30stream.log 2>&1 && echo '=== 30B-STREAM SPLIT OK ===' >> $LOG || { echo '=== 30B-STREAM SPLIT FALLO ===' >> $LOG; exit 1 }
$PY -u src/serve.py --artifact artifacts/qwen30-stream --pool-k 32 --refresh 256 --max-tokens 256 > /tmp/s30.txt 2>&1
grep -E "tok/s|peak|GB" /tmp/s30.txt | tail -3 >> $LOG; rm -f /tmp/s30.txt
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench math500,mbpp --n 100 --tag b30s_hard >> runs/bench30s.log 2>&1 && echo '=== 30B-STREAM HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen30-stream --stage bench --bench mmlu,lambada --n 500 --tag b30s_wide >> runs/bench30s.log 2>&1 && echo '=== 30B-STREAM WIDE OK ===' >> $LOG
echo "=== 30B-STREAM COMPLETO $(date) ===" >> $LOG
