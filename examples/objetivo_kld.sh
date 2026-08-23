#!/bin/zsh
# OBJETIVO: superar la KLD del 2-bit estático. (1) política absorb (el shared
# absorbe la masa caída) en GPU; (2) KLD del rival UD-Q2 contra NUESTRA base
# exacta (misma referencia), via llama-server en CPU. Secuencial, con humo.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
GGUF=artifacts/unsloth/Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_XL.gguf
until grep -q "BENCH-C290 COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|src/serve.py" > /dev/null; do sleep 60; done
# --- rival contra la misma referencia ---
echo "=== RIVAL-KLD START $(date) ===" >> $LOG
llama-server -m $GGUF -ngl 0 -c 4096 --port 8080 --host 127.0.0.1 > runs/rival_kld_server.log 2>&1 &
SRV=$!
for i in $(seq 1 180); do curl -s http://127.0.0.1:8080/health | grep -q '"ok"' && break; sleep 5; kill -0 $SRV 2>/dev/null || { echo '=== RIVAL-KLD SERVER MUERTO ===' >> $LOG; exit 1; }; done
curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' -d '{"prompt":[151644,872],"n_predict":1,"n_probs":5,"temperature":0}' > runs/rival_kld_smoke.json
grep -q "completion_probabilities" runs/rival_kld_smoke.json || { echo '=== RIVAL-KLD HUMO FALLO ===' >> $LOG; kill $SRV; exit 1; }
$PY -u src/eval_stream.py --artifact $A --stage kldremote --openai-base http://127.0.0.1:8080 --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag udq2_vs_base >> runs/rival_kld.log 2>&1 && echo '=== RIVAL-KLD OK ===' >> $LOG
kill $SRV 2>/dev/null
echo "=== RIVAL-KLD COMPLETO $(date) ===" >> $LOG
