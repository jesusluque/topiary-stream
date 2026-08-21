#!/bin/zsh
# KLD servido-vs-base + divergencia de trayectoria (80B) — 4 pasadas, código
# con el fix de corpus. Sin gate: lanzado con GPU libre.
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream
ORD=artifacts/qwen80-stream/orders_routed.npz
echo "=== KLD FULL START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --serve-mode exact --orders $ORD --data-general data/lambada.jsonl --chunks-general 30 --kld-out runs/kld80_base.npz --tag k80base >> runs/kld80.log 2>&1 && echo '=== KLD BASE OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general data/lambada.jsonl --chunks-general 30 --kld-ref runs/kld80_base.npz --tag k80served >> runs/kld80.log 2>&1 && echo '=== KLD SERVIDO OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage traj --serve-mode exact --orders $ORD --n 300 --gen-len 32 --kld-out runs/traj80_base.npz --tag t80base >> runs/kld80.log 2>&1 && echo '=== TRAJ BASE OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage traj --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --n 300 --gen-len 32 --kld-ref runs/traj80_base.npz --tag t80served >> runs/kld80.log 2>&1 && echo '=== TRAJ SERVIDO OK ===' >> $LOG
echo "=== KLD-TRAJ COMPLETO $(date) ===" >> $LOG
