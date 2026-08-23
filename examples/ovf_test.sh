#!/bin/zsh
# Test del refresh barato (tier de desbordamiento): humo → velocidad → KLD
# a cadencia 32 con merge cada 8. Comparar con r32: KLD 0.303 @ 9.1 tok/s.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
until grep -qE "RIVAL-KLD (COMPLETO|SERVER MUERTO|HUMO FALLO)" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== OVF START $(date) ===" >> $LOG
$PY -u src/serve.py --artifact $A --pool-c 240 --pool-k 32 --orders $ORD --refresh 32 --ovf-merge 8 --max-tokens 64 > runs/ovf_smoke.txt 2>&1
grep -q "tok/s" runs/ovf_smoke.txt || { echo '=== OVF HUMO FALLO ===' >> $LOG; exit 1; }
$PY -u src/serve.py --artifact $A --pool-c 240 --pool-k 32 --orders $ORD --refresh 32 --ovf-merge 8 --max-tokens 256 2>&1 | grep "tok/s" | sed 's/^/[speed ovf r32] /' >> runs/campana.log
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --kld-refresh 32 --ovf-merge 8 --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80ovf_r32 >> runs/kld80.log 2>&1 && echo '=== OVF KLD OK ===' >> $LOG
echo "=== OVF COMPLETO $(date) ===" >> $LOG
