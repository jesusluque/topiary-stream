#!/bin/zsh
# EL DUELO: Unsloth UD-Q2_K_XL del 80B en llama.cpp (Metal, mmap — no cabe
# entero en 24 GB) vs nuestro 80B Stream (C=240). Misma máquina, misma
# batería, mismos prompts. Gate: GPU libre tras KLD-decode y f48.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
GGUF=artifacts/unsloth/Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_XL.gguf
until grep -q "KLD-DECODE COMPLETO" runs/ablation.log 2>/dev/null && grep -q "F48 OK" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py|f48_persistence|llama-" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== DUEL START $(date) ===" >> $LOG
sysctl vm.swapusage >> runs/duel.log
# 1) velocidad: llama-bench, 1024 tokens de generación, Metal
llama-bench -m $GGUF -ngl 99 -p 0 -n 1024 -r 2 >> runs/duel.log 2>&1 && echo '=== DUEL BENCH OK ===' >> $LOG
sysctl vm.swapusage >> runs/duel.log
# 2) calidad: llama-server + nuestra batería via API (mismos prompts/parsers)
llama-server -m $GGUF -ngl 99 -c 4096 --port 8080 --host 127.0.0.1 > runs/duel_server.log 2>&1 &
SRV=$!
until curl -s http://127.0.0.1:8080/health | grep -q '"ok"'; do sleep 5; kill -0 $SRV 2>/dev/null || { echo '=== DUEL SERVER MUERTO ===' >> $LOG; exit 1; }; done
$PY -u src/eval_stream.py --artifact artifacts/qwen80-stream --openai-base http://127.0.0.1:8080 --stage bench --bench math500,mbpp --n 100 --tag udq2_hard >> runs/duel.log 2>&1 && echo '=== DUEL HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen80-stream --openai-base http://127.0.0.1:8080 --stage bench --bench mmlu,lambada --n 500 --tag udq2_wide >> runs/duel.log 2>&1 && echo '=== DUEL WIDE OK ===' >> $LOG
kill $SRV 2>/dev/null
sysctl vm.swapusage >> runs/duel.log
echo "=== DUEL COMPLETO $(date) ===" >> $LOG
