#!/bin/zsh
# Persistencia token-a-token del routing del 235B (f48, lab) en cuanto exista
# el artefacto y el GPU esté libre. Decide: pool de seguimiento vs
# autoespeculación sobre suelo.
LOG=$(dirname "$0")/../runs/ablation.log
until grep -q "ARTEFACTO COMPLETO" $(dirname "$0")/../runs/rebuild235.log 2>/dev/null && grep -q "KLD-DECODE COMPLETO" $LOG 2>/dev/null && ! pgrep -f "eval_stream|src/serve.py|llama-" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream" > /dev/null && exec "$0"
cd ${LAB:-$HOME/luc/nanite-moe}
echo "=== F48 START $(date) ===" >> $LOG
caffeinate -is .venv/bin/python -u src/f48_persistence.py --artifact ../topiary-stream/artifacts/qwen235-stream --chunks 6 --chunk-len 512 > runs/f48_235b.log 2>&1 && echo '=== F48 OK ===' >> $LOG || echo '=== F48 FALLO ===' >> $LOG
