#!/bin/zsh
# Repetir la pasada servida C=240 (produccion) con el codigo que guarda la
# curva por posicion: ¿falla desde el token 65 o se acumula?
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=/Users/muriel/luc/nanite-moe/data/calib_general_qwen3/held_out.jsonl
until grep -q "KLD-C340 COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|f48_persistence" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== KLD-C240-CURVA START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80c240_curve >> runs/kld80.log 2>&1 && echo '=== KLD-C240-CURVA OK ===' >> $LOG
echo "=== KLD-C240-CURVA COMPLETO $(date) ===" >> $LOG
