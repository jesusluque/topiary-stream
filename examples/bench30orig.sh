#!/bin/zsh
# Control: el Qwen3-30B-A3B-4bit ORIGINAL (sin esculpir) en los mismos 4 ejes.
# Espera a: (1) descarga completa, (2) GPU libre (ni evals ni ablación).
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
LOG=runs/ablation.log
until grep -q "30B ORIGINAL DESCARGADO" runs/dl30orig.log 2>/dev/null \
      && grep -q "SOLUCIONES OK" runs/ablation.log 2>/dev/null \
      && ! pgrep -f "eval_stream|src/serve.py" > /dev/null; do
  sleep 120
done
M=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen3-30B-A3B-4bit'))")
echo "=== ORIGINAL START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $M --stage bench --bench math500,mbpp --n 100 --tag b30orig_hard >> runs/bench30orig.log 2>&1 && echo '=== 30B-ORIG HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $M --stage bench --bench mmlu,lambada --n 500 --tag b30orig_wide >> runs/bench30orig.log 2>&1 && echo '=== 30B-ORIG WIDE OK ===' >> $LOG
echo "--- 30B_original_nativo" >> $LOG
$PY -m mlx_lm generate --model $M --prompt "Write a detailed essay about the history of computing." --max-tokens 256 2>&1 | grep -E "tokens-per-sec|Peak" >> $LOG
echo "=== ORIGINAL OK $(date) ===" >> $LOG
