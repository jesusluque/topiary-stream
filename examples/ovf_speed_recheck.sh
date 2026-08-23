#!/bin/zsh
# Re-medida LIMPIA de velocidad (el 5.7 del OVF se midió con 3-6 GB de swap
# por la carrera del llama-server): OVF r32 vs r32 plano vs r256, 256 tokens.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
until grep -q "OVF COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|src/serve.py" > /dev/null; do sleep 20; done
echo "=== SPEED-RECHECK START $(date) swap=$(sysctl -n vm.swapusage | awk '{print $6}') ===" >> $LOG
for c in "ovf_r32 --refresh 32 --ovf-merge 8" "r32 --refresh 32" "r256 --refresh 256"; do
  set -- ${=c}; name=$1; shift
  $PY -u src/serve.py --artifact $A --pool-c 240 --pool-k 32 --orders $ORD "$@" --max-tokens 256 2>&1 | grep "tok/s" | sed "s/^/[speed2 $name] /" >> runs/campana.log
done
echo "=== SPEED-RECHECK COMPLETO $(date) ===" >> $LOG
