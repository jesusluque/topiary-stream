#!/bin/zsh
# KLD-decode REALISTA: chunks de 512 tokens (held-out wiki del lab), prefijo
# exacto de 64 tokens → el pool se calienta con contexto real, como en
# producción. Contraste con la variante LAMBADA (80 tokens, prefijo 16,
# cambio de tema cada chunk: escenario pesimista). Gate: tras el duelo.
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=/Users/muriel/luc/nanite-moe/data/calib_general_qwen3/held_out.jsonl
until grep -q "DUEL COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|f48_persistence|llama-" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|f48_persistence|llama-" > /dev/null && exec "$0"
echo "=== KLD-LONG START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 1 --chunk-len 128 --kld-out runs/kld_smoke.npz --tag smoke >> runs/kld80.log 2>&1
[ -f runs/kld_smoke.npz ] || { echo '=== KLD-LONG HUMO FALLO ===' >> $LOG; exit 1; }; rm -f runs/kld_smoke.npz
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode exact --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-out runs/kld80_base_long.npz --tag k80base_long >> runs/kld80.log 2>&1 && echo '=== KLD-LONG BASE OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80served_long >> runs/kld80.log 2>&1 && echo '=== KLD-LONG SERVIDO OK ===' >> $LOG
echo "=== KLD-LONG COMPLETO $(date) ===" >> $LOG
