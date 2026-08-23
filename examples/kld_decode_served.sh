#!/bin/zsh
# Solo la pasada SERVIDA de la KLD-decode (la referencia base ya existe).
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
[ -f runs/kld80_base_decode.npz ] || { echo '=== KLD-DECODE SIN REFERENCIA ===' >> $LOG; exit 1; }
echo "=== KLD-DECODE SERVIDO START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general data/lambada.jsonl --chunks-general 30 --kld-ref runs/kld80_base_decode.npz --tag k80served_dec >> runs/kld80.log 2>&1 && echo '=== KLD-DECODE SERVIDO OK ===' >> $LOG
echo "=== KLD-DECODE COMPLETO $(date) ===" >> $LOG
