#!/bin/zsh
# DUELO v3 — última config honesta de llama.cpp: TODO en CPU (-ngl 0, mmap),
# el modo que llama.cpp usa cuando el modelo no cabe en la GPU. Gate: tras
# KLD-FLOOR, máquina libre (compite por RAM con nuestros memmaps).
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
GGUF=artifacts/unsloth/Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_XL.gguf
until grep -q "KLD-FLOOR COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py|f48_persistence|llama-|floor.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== DUEL3 START $(date) ===" >> $LOG
sysctl vm.swapusage >> runs/duel3.log
llama-bench -m $GGUF -ngl 0 -p 0 -n 128 -r 2 >> runs/duel3.log 2>&1 && echo '=== DUEL3 BENCH OK ===' >> $LOG || echo '=== DUEL3 BENCH FALLO ===' >> $LOG
sysctl vm.swapusage >> runs/duel3.log
llama-server -m $GGUF -ngl 0 -c 4096 --port 8080 --host 127.0.0.1 > runs/duel3_server.log 2>&1 &
SRV=$!
for i in $(seq 1 180); do curl -s http://127.0.0.1:8080/health | grep -q '"ok"' && break; sleep 5; kill -0 $SRV 2>/dev/null || { echo '=== DUEL3 SERVER MUERTO ===' >> $LOG; exit 1; }; done
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}' > runs/duel3_smoke.json 2>&1
grep -q '"content"' runs/duel3_smoke.json || { echo '=== DUEL3 HUMO FALLO ===' >> $LOG; kill $SRV; exit 1; }
$PY -u src/eval_stream.py --artifact artifacts/qwen80-stream --openai-base http://127.0.0.1:8080 --stage bench --bench math500,mbpp --n 100 --tag udq2_hard >> runs/duel3.log 2>&1 && echo '=== DUEL3 HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen80-stream --openai-base http://127.0.0.1:8080 --stage bench --bench mmlu --n 500 --tag udq2_wide >> runs/duel3.log 2>&1 && echo '=== DUEL3 WIDE OK ===' >> $LOG
kill $SRV 2>/dev/null
sysctl vm.swapusage >> runs/duel3.log
echo "=== DUEL3 COMPLETO $(date) ===" >> $LOG
