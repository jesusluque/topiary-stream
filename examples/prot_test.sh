#!/bin/zsh
# Test de protección de tensores (Unsloth dentro de la pirámide):
#  prot  = router BF16;  prot8 = router BF16 + esqueleto no-experto a 8 bits.
# Por artefacto: base exacta propia (KLD vs sí mismo), servido r128, y
# batería hard n=100 (sin referencia: MATH/MBPP vs 65/81 de producción).
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
D=/Users/muriel/luc/nanite-moe/data/calib_general_qwen3/held_out.jsonl
until grep -q "NOCHE567 COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
for P in prot prot8; do
  A=artifacts/qwen80-$P; ORD=$A/orders_routed.npz
  [ -f $A/model.safetensors ] || { echo "=== $P AUSENTE ===" >> $LOG; continue; }
  echo "=== PROT-$P START $(date) ===" >> $LOG
  $PY -u src/serve.py --artifact $A --pool-c 240 --pool-k 32 --orders $ORD --max-tokens 64 > runs/prot_${P}_smoke.txt 2>&1
  grep -q "tok/s" runs/prot_${P}_smoke.txt || { echo "=== PROT-$P HUMO FALLO ===" >> $LOG; continue; }
  $PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode exact --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-out runs/kld80_base_${P}.npz --tag k80base_$P >> runs/prot.log 2>&1 && echo "=== PROT-$P BASE OK ===" >> $LOG
  $PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --kld-refresh 128 --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_${P}.npz --tag k80served_$P >> runs/prot.log 2>&1 && echo "=== PROT-$P KLD OK ===" >> $LOG
  $PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 100 --serve-mode nosync --gen-refresh 128 --pool-c 240 --pool-k 32 --orders $ORD --tag c80${P}_hard >> runs/prot.log 2>&1 && echo "=== PROT-$P HARD OK ===" >> $LOG
done
echo "=== PROT COMPLETO $(date) ===" >> $LOG
