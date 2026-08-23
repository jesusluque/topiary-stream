#!/bin/zsh
# "Modo 2-bit dinámico" (idea LOD del usuario): K→0 para gastar los bytes en
# cobertura: C=290 (57%, el maximo que cabe: C=340 cargaba 20.7GB y hacia thrashing) todo a P0. ¿Baja la KLD realista desde 0.774?
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
until ! pgrep -f "eval_stream|llama-|f48_persistence" > /dev/null; do sleep 30; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== KLD-C340 START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode nosync --pool-c 290 --pool-k 1 --orders $ORD --data-general $D --chunks-general 1 --chunk-len 128 --kld-out runs/kld_smoke.npz --tag smoke >> runs/kld80.log 2>&1
[ -f runs/kld_smoke.npz ] || { echo '=== KLD-C340 HUMO FALLO ===' >> $LOG; exit 1; }; rm -f runs/kld_smoke.npz
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode nosync --pool-c 290 --pool-k 1 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80c340_long >> runs/kld80.log 2>&1 && echo '=== KLD-C340 OK ===' >> $LOG
echo "=== KLD-C340 COMPLETO $(date) ===" >> $LOG
