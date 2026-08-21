#!/bin/zsh
# KLD en régimen de DECODE (la métrica real del peaje del pool), tras las traj.
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream
ORD=artifacts/qwen80-stream/orders_routed.npz
until grep -q "KLD-TRAJ COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|f48_persistence|llama-" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|f48_persistence|llama-" > /dev/null && exec "$0"
echo "=== KLD-DECODE START $(date) ===" >> $LOG
# HUMO (2 min) antes de horas: 1 chunk, y el npz debe existir. Dos bugs
# silenciosos (n_tokens, shadowing de `out`) costaron ~1.5h de GPU cada uno.
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general data/lambada.jsonl --chunks-general 1 --kld-out runs/kld_smoke.npz --tag smoke >> runs/kld80.log 2>&1
[ -f runs/kld_smoke.npz ] || { echo '=== KLD-DECODE HUMO FALLO ===' >> $LOG; exit 1; }
rm -f runs/kld_smoke.npz
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode exact --orders $ORD --data-general data/lambada.jsonl --chunks-general 30 --kld-out runs/kld80_base_decode.npz --tag k80base_dec >> runs/kld80.log 2>&1 && echo '=== KLD-DECODE BASE OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general data/lambada.jsonl --chunks-general 30 --kld-ref runs/kld80_base_decode.npz --tag k80served_dec >> runs/kld80.log 2>&1 && echo '=== KLD-DECODE SERVIDO OK ===' >> $LOG
echo "=== KLD-DECODE COMPLETO $(date) ===" >> $LOG
