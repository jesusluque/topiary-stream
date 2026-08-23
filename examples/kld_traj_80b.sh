#!/bin/zsh
# KLD servido-vs-base + divergencia de trayectoria (80B): la métrica del
# revisor hostil (2407.09141). Base = modo exact (el control que ya tenemos).
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream
ORD=artifacts/qwen80-stream/orders_routed.npz
until grep -q "OPT30 COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 20 + 5)); pgrep -f "eval_stream" > /dev/null && exec "$0"
echo "=== KLD START $(date) ===" >> $LOG
# 1) KLD teacher-forced sobre LAMBADA held-out (30 chunks)
$PY -u src/eval_stream.py --artifact $A --stage kld --serve-mode exact --orders $ORD --data-general data/lambada.jsonl --chunks-general 30 --kld-out runs/kld80_base.npz --tag k80base >> runs/kld80.log 2>&1 && echo '=== KLD BASE OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general data/lambada.jsonl --chunks-general 30 --kld-ref runs/kld80_base.npz --tag k80served >> runs/kld80.log 2>&1 && echo '=== KLD SERVIDO OK ===' >> $LOG
# 2) Divergencia de trayectoria greedy 300@32 (base exacta -> servido)
$PY -u src/eval_stream.py --artifact $A --stage traj --serve-mode exact --orders $ORD --n 300 --gen-len 32 --kld-out runs/traj80_base.npz --tag t80base >> runs/kld80.log 2>&1 && echo '=== TRAJ BASE OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage traj --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --n 300 --gen-len 32 --kld-ref runs/traj80_base.npz --tag t80served >> runs/kld80.log 2>&1 && echo '=== TRAJ SERVIDO OK ===' >> $LOG
echo "=== KLD-TRAJ COMPLETO $(date) ===" >> $LOG
