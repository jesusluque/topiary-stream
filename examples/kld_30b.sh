#!/bin/zsh
# KLD-decode realista en el 30B-Stream (suelo universal para TODOS): si el
# invariante del suelo es lo que importa, aquí la KLD debe salir baja.
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
A=artifacts/qwen30-stream
D=/Users/muriel/luc/nanite-moe/data/calib_general_qwen3/held_out.jsonl
until grep -qE "DUEL3 (COMPLETO|SERVER MUERTO|HUMO FALLO)" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|f48_persistence" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== KLD-30B START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --pool-k 32 --data-general $D --chunks-general 1 --chunk-len 128 --kld-out runs/kld_smoke.npz --tag smoke >> runs/kld30.log 2>&1
[ -f runs/kld_smoke.npz ] || { echo '=== KLD-30B HUMO FALLO ===' >> $LOG; exit 1; }; rm -f runs/kld_smoke.npz
# base = el MISMO checkpoint nativo (taper) sin runtime, token a token
$PY -u src/eval_stream.py --artifact /Users/muriel/luc/nanite-moe/models/qwen3-30b-4bit-fine-taper085 --stage kld --kld-decode --data-general $D --chunks-general 4 --chunk-len 512 --kld-out runs/kld30_base_long.npz --tag k30base_long >> runs/kld30.log 2>&1 && echo '=== KLD-30B BASE OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --pool-k 32 --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld30_base_long.npz --tag k30served_long >> runs/kld30.log 2>&1 && echo '=== KLD-30B SERVIDO OK ===' >> $LOG
echo "=== KLD-30B COMPLETO $(date) ===" >> $LOG
