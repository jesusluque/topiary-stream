#!/bin/zsh
# Generalidad DeepSeek-V2-Lite, segunda cadena: CONTROLES primero.
#  (1) KLD exacto-vs-base (mismo camino → ≈0: valida el pipeline en esta familia)
#  (2) MATH/MBPP exactos n=30 (¿el modelo produce \boxed? ¿o el 0/40 de C=64 es suyo?)
#  (3) C=64 con K=64 (todo P1 → ≈0, control) y K=32; luego C=30/15 con K=16
#  (4) baterías n=100 / n=500 para exacto y cada C
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log; OUT=runs/generality_dsv2.log
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
A=artifacts/dsv2lite-stream; ORD=$A/orders_routed.npz; REF=runs/kldds_base_long.npz
until ! pgrep -f "eval_stream|src/serve.py" > /dev/null; do sleep 30; done
echo "=== GEN-DSV2-B START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --serve-mode exact --kld-decode --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref $REF --tag ds_exact_ctrl >> $OUT 2>&1 && echo "=== GEN-DSV2-B KLD EXACT-CTRL: $(python3 -c "import json;d=json.load(open('runs/kld_ds_exact_ctrl.json'));print(round(d['kld_mean'],5), 'p99', round(d['kld_p99'],4))") ===" >> $LOG || echo "=== GEN-DSV2-B KLD EXACT-CTRL FALLO ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 30 --serve-mode exact --tag ds_exact_n30 >> $OUT 2>&1 && echo "=== GEN-DSV2-B EXACT n30: $(python3 -c "import json;d=json.load(open('runs/bench_ds_exact_n30.json'));print({k:v for k,v in d.items() if k in ('math500','mbpp')})") ===" >> $LOG || echo "=== GEN-DSV2-B EXACT n30 FALLO ===" >> $LOG
for CK in 64:64 64:32; do C=${CK%:*}; K=${CK#*:}
  $PY -u src/eval_stream.py --artifact $A --stage kld --serve-mode nosync --kld-decode --kld-refresh 256 --pool-c $C --pool-k $K --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref $REF --tag ds_c${C}k${K} >> $OUT 2>&1 && echo "=== GEN-DSV2-B KLD C=$C K=$K: $(python3 -c "import json;d=json.load(open('runs/kld_ds_c${C}k${K}.json'));print(round(d['kld_mean'],4), 'p99', round(d['kld_p99'],3))") ===" >> $LOG || echo "=== GEN-DSV2-B KLD C=$C K=$K FALLO ===" >> $LOG
done
for CK in 30:16 15:16; do C=${CK%:*}; K=${CK#*:}
  $PY -u src/eval_stream.py --artifact $A --stage kld --serve-mode nosync --kld-decode --kld-refresh 256 --pool-c $C --pool-k $K --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref $REF --tag ds_c${C}k${K} >> $OUT 2>&1 && echo "=== GEN-DSV2-B KLD C=$C K=$K: $(python3 -c "import json;d=json.load(open('runs/kld_ds_c${C}k${K}.json'));print(round(d['kld_mean'],4), 'p99', round(d['kld_p99'],3))") ===" >> $LOG || echo "=== GEN-DSV2-B KLD C=$C K=$K FALLO ===" >> $LOG
done
# baterías: exacto y C=64/K=32, C=30/K=16, C=15/K=16
$PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 100 --serve-mode exact --tag ds_exact_hard >> $OUT 2>&1 && echo "=== GEN-DSV2-B HARD EXACT OK ===" >> $LOG || echo "=== GEN-DSV2-B HARD EXACT FALLO ===" >> $LOG
for CK in 64:32 30:16 15:16; do C=${CK%:*}; K=${CK#*:}
  $PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 100 --serve-mode nosync --gen-refresh 128 --pool-c $C --pool-k $K --orders $ORD --tag ds_c${C}k${K}_hard >> $OUT 2>&1 && echo "=== GEN-DSV2-B HARD C=$C K=$K OK ===" >> $LOG || echo "=== GEN-DSV2-B HARD C=$C K=$K FALLO ===" >> $LOG
done
$PY -u src/eval_stream.py --artifact $A --stage bench --bench mmlu,lambada --n 500 --serve-mode exact --tag ds_exact_wide >> $OUT 2>&1 && echo "=== GEN-DSV2-B WIDE EXACT OK ===" >> $LOG || echo "=== GEN-DSV2-B WIDE EXACT FALLO ===" >> $LOG
for CK in 64:32 30:16 15:16; do C=${CK%:*}; K=${CK#*:}
  $PY -u src/eval_stream.py --artifact $A --stage bench --bench mmlu,lambada --n 500 --serve-mode nosync --gen-refresh 128 --pool-c $C --pool-k $K --orders $ORD --tag ds_c${C}k${K}_wide >> $OUT 2>&1 && echo "=== GEN-DSV2-B WIDE C=$C K=$K OK ===" >> $LOG || echo "=== GEN-DSV2-B WIDE C=$C K=$K FALLO ===" >> $LOG
done
echo "=== GEN-DSV2-B COMPLETO $(date) ===" >> $LOG
