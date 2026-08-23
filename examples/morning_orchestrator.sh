#!/bin/zsh
# Orquestador post-reinicio (2026-08-20): bench citable en frío PRIMERO,
# luego reanudar todo lo que el reboot mató.
cd "$(dirname "$0")/.."
export PATH=${LAB:-$HOME/luc/nanite-moe}/.venv/bin:$PATH
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log

# 1. Esperar asentamiento: >=10 min de uptime y load < 4
while true; do
  up=$(sysctl -n kern.boottime | awk '{print $4}' | tr -d ',')
  age=$(( $(date +%s) - up ))
  load=$(sysctl -n vm.loadavg | awk '{print int($2)}')
  [ $age -ge 600 ] && [ $load -lt 4 ] && break
  sleep 30
done

# 2. Bench citable en frío (la oportunidad del reboot)
./examples/citable_bench.sh > runs/citable_20260820.log 2>&1
echo "=== CITABLE OK $(date) ===" >> $LOG

# 3. Reanudar descarga 235B (red; ya no molesta al bench)
nohup caffeinate -is zsh -c 'hf download mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit-DWQ >/dev/null 2>&1 && echo "=== DESCARGA OK ===" && python -u src/split.py --src mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit-DWQ --out artifacts/qwen235-stream --layout full-memmap --consume && echo "=== SPLIT OK ===" && cp artifacts/qwen235-stream-kit/floor256.safetensors artifacts/qwen235-stream/ && cp artifacts/qwen235-stream-kit/orders_routed.npz artifacts/qwen235-stream/ && echo "=== ARTEFACTO COMPLETO ==="' >> runs/rebuild235.log 2>&1 &!

# 4. Re-lanzar el wide del C=120 (murió a mitad con el reboot)
python -u src/eval_stream.py --artifact artifacts/qwen80-stream --stage bench --bench mmlu,lambada --n 500 --pool-c 120 --pool-k 32 --orders artifacts/qwen80-stream/orders_routed.npz --tag b80_c120_wide >> runs/c120.log 2>&1 && echo '=== C120 WIDE OK ===' >> $LOG
echo "=== C120 COMPLETO $(date) ===" >> $LOG

# 5. Batería opt30 (K4/K64): re-lanzar su vigía, que ahora sí verá el marcador
nohup caffeinate -is ./examples/opt30_full.sh > /dev/null 2>&1 &!
echo "=== ORQUESTADOR OK $(date) ===" >> $LOG
