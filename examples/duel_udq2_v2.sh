#!/bin/zsh
# DUELO v2: el UD-Q2_K_XL (30GB) con -ngl 99 muere por OOM de Metal en 24GB
# (medido: kIOGPUCommandBufferCallbackErrorOutOfMemory). Config justa de
# llama.cpp para MoE > RAM: --cpu-moe (expertos en RAM/mmap, atención en
# Metal). Mismo protocolo: bench 1024 tok + batería por API. Gate: tras
# KLD-LONG, GPU libre.
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
GGUF=artifacts/unsloth/Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_XL.gguf
until grep -q "KLD-LONG COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py|f48_persistence|llama-" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== DUEL2 START $(date) ===" >> $LOG
sysctl vm.swapusage >> runs/duel2.log
llama-bench -m $GGUF -ngl 99 -cmoe 1 -p 0 -n 256 -r 2 >> runs/duel2.log 2>&1 && echo '=== DUEL2 BENCH OK ===' >> $LOG || echo '=== DUEL2 BENCH FALLO ===' >> $LOG
sysctl vm.swapusage >> runs/duel2.log
llama-server -m $GGUF -ngl 99 --cpu-moe -c 4096 --port 8080 --host 127.0.0.1 > runs/duel2_server.log 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -s http://127.0.0.1:8080/health | grep -q '"ok"' && break; sleep 5; kill -0 $SRV 2>/dev/null || { echo '=== DUEL2 SERVER MUERTO ===' >> $LOG; exit 1; }; done
# humo: una petición real antes de la batería
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}' > runs/duel2_smoke.json 2>&1
grep -q '"content"' runs/duel2_smoke.json || { echo '=== DUEL2 HUMO FALLO ===' >> $LOG; kill $SRV; exit 1; }
$PY -u src/eval_stream.py --artifact artifacts/qwen80-stream --openai-base http://127.0.0.1:8080 --stage bench --bench math500,mbpp --n 100 --tag udq2_hard >> runs/duel2.log 2>&1 && echo '=== DUEL2 HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact artifacts/qwen80-stream --openai-base http://127.0.0.1:8080 --stage bench --bench mmlu --n 500 --tag udq2_wide >> runs/duel2.log 2>&1 && echo '=== DUEL2 WIDE OK ===' >> $LOG
kill $SRV 2>/dev/null
sysctl vm.swapusage >> runs/duel2.log
echo "=== DUEL2 COMPLETO $(date) ===" >> $LOG
