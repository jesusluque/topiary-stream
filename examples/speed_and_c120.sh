#!/bin/zsh
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
ORIG=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen3-30B-A3B-4bit'))")
TAPER=/Users/muriel/luc/nanite-moe/models/qwen3-30b-4bit-fine-taper085
echo "=== SPEED NATIVO START $(date) ===" >> $LOG
for round in 1 2; do
  $PY -u examples/speed_nativo.py $ORIG  original_r$round >> $LOG 2>&1
  $PY -u examples/speed_nativo.py $TAPER taper_r$round    >> $LOG 2>&1
done
echo "=== SPEED NATIVO OK ===" >> $LOG
echo "=== C120 QUALITY START $(date) ===" >> $LOG
A80="artifacts/qwen80-stream"
$PY -u src/eval_stream.py --artifact $A80 --stage bench --bench math500,mbpp --n 100 --pool-c 120 --pool-k 32 --orders $A80/orders_routed.npz --tag b80_c120_hard >> runs/c120.log 2>&1 && echo '=== C120 HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A80 --stage bench --bench mmlu,lambada --n 500 --pool-c 120 --pool-k 32 --orders $A80/orders_routed.npz --tag b80_c120_wide >> runs/c120.log 2>&1 && echo '=== C120 WIDE OK ===' >> $LOG
echo "=== C120 COMPLETO $(date) ===" >> $LOG
