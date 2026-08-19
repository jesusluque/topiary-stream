#!/bin/zsh
# Banco de las SOLUCIONES (v2, fallos de zsh corregidos): calidad del campeón
# Topiary + matriz de velocidad de las soluciones y sus variaciones.
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src   # dense_loader (taper per-layer)
LOG=runs/ablation.log
M30=/Users/muriel/luc/nanite-moe/models/qwen3-30b-4bit-fine-taper085

echo "=== SOLUCIONES v2 START $(date) ===" >> $LOG

# FASE 1 — calidad del campeón Topiary estático (30B) en los mismos 4 ejes
$PY -u src/eval_stream.py --artifact $M30 --stage bench --bench math500,mbpp --n 100 --tag b30_hard >> runs/bench30.log 2>&1 && echo '=== 30B HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $M30 --stage bench --bench mmlu,lambada --n 500 --tag b30_wide >> runs/bench30.log 2>&1 && echo '=== 30B WIDE OK ===' >> $LOG

# FASE 2 — velocidad (${=VAR} fuerza el word-split en zsh)
run() {
  echo "--- $1" >> $LOG
  local name=$1; shift
  $PY -u src/serve.py --max-tokens 256 ${=@} >> /tmp/serve_out.txt 2>&1
  grep -E "tok/s|peak" /tmp/serve_out.txt | tail -2 >> $LOG
  rm -f /tmp/serve_out.txt
}
echo "--- 30B_topiary_nativo" >> $LOG
$PY -u -m mlx_lm generate --model $M30 --prompt "Write a detailed essay about the history of computing." --max-tokens 256 2>&1 | grep -iE "tokens-per-sec|peak" >> $LOG

A35="--artifact artifacts/qwen35-stream"
A80="--artifact artifacts/qwen80-stream --orders artifacts/qwen80-stream/orders_routed.npz"
run "35B_baseline_K32_r256"  "$A35 --pool-k 32 --refresh 256"
run "35B_K4"                 "$A35 --pool-k 4  --refresh 256"
run "35B_K64"                "$A35 --pool-k 64 --refresh 256"
run "35B_refresh64"          "$A35 --pool-k 32 --refresh 64"
run "35B_governor"           "$A35 --pool-k 32 --refresh 256 --governor"
run "80B_baseline_C240_K32"  "$A80 --pool-c 240 --pool-k 32 --refresh 256"
run "80B_C120"               "$A80 --pool-c 120 --pool-k 32 --refresh 256"
run "80B_refresh64"          "$A80 --pool-c 240 --pool-k 32 --refresh 64"

echo "=== SOLUCIONES v2 OK $(date) ===" >> $LOG
