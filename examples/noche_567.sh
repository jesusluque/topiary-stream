#!/bin/zsh
# Ítems 6, 7 y 5 de la noche, tras la batería C=290:
#  6) ráfaga tras el prompt (KLD, con OVF si pasó);  7) gobernador de dos
#  marchas (KLD wiki + MATH n=50);  5) n>=300 en MATH/MBPP con la config
#  elegida por selector (mejor n=100 de C240-r128 / C290; +OVF y ráfaga si rentan).
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
REF=runs/kld80_base_long.npz
until grep -q "BENCH-C290 COMPLETO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-|src/serve.py" > /dev/null; do sleep 60; done
sleep $((RANDOM % 15 + 5)); pgrep -f "eval_stream|llama-" > /dev/null && exec "$0"
echo "=== NOCHE567 START $(date) ===" >> $LOG

# ¿pasó el refresh barato? (KLD <= 0.35 y tok/s >= 12.6) → cadencia 32 + OVF
OVFOPT="--kld-refresh 128"; GENR=128; OVFGEN=""
if [ -f runs/kld_k80ovf_r32.json ]; then
  KL=$($PY -c "import json;print(json.load(open('runs/kld_k80ovf_r32.json'))['kld_mean'])")
  SP=$(grep -oE "\[speed ovf r32\].*= ([0-9.]+) tok/s" runs/campana.log | grep -oE "[0-9.]+ tok/s" | cut -d' ' -f1 | tail -1)
  if $PY -c "import sys; sys.exit(0 if float('$KL')<=0.35 and float('${SP:-0}')>=12.6 else 1)"; then
    OVFOPT="--kld-refresh 32 --ovf-merge 8"; GENR=32; OVFGEN="--ovf-merge 8"; echo "=== OVF ADOPTADO (KLD $KL @ $SP tok/s) ===" >> $LOG
  else echo "=== OVF NO ADOPTADO (KLD $KL @ ${SP:-?} tok/s) ===" >> $LOG; fi
fi

# 6) ráfaga tras el prompt: KLD wiki con burst 64/8
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode ${=OVFOPT} --burst-len 64 --burst-every 8 --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref $REF --tag k80burst >> runs/kld80.log 2>&1 && echo '=== BURST KLD OK ===' >> $LOG

# 7) gobernador de dos marchas: KLD wiki (debe subir a marcha hi) + MATH n=50 (debe quedarse en lo)
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode ${=OVFOPT} --gear 240:32,290:1 --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref $REF --tag k80gear >> runs/kld80.log 2>&1 && echo '=== GEAR KLD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench math500 --n 50 --gear 240:32,290:1 --gen-refresh $GENR ${=OVFGEN} --serve-mode nosync --pool-c 240 --pool-k 32 --orders $ORD --tag c80gear_math50 >> runs/campana.log 2>&1 && echo '=== GEAR MATH50 OK ===' >> $LOG

# 5) selector: mejor (math+mbpp) n=100 entre C240-r128 y C290; ráfaga si mejoró el arranque
$PY - <<'PYEOF' >> runs/campana.log 2>&1
import json
def s(tag):
    try: d=json.load(open(f"runs/bench_{tag}.json")); return d["math500"]+d["mbpp"]
    except Exception: return -1
best = "c290" if s("c80c290_hard") > s("c80best_hard") else "c240"
burst = False
try:
    kb = json.load(open("runs/kld_k80burst.json"))["by_position"]["pos_0-64"]
    kr = json.load(open("runs/kld_cand_r128.json"))["by_position"]["pos_0-64"]
    burst = kb < kr * 0.85
except Exception: pass
print("SELECTOR n300:", best, "| burst:", burst, "| scores:", s("c80c290_hard"), s("c80best_hard"))
open("runs/n300_cfg.txt","w").write(f"{best} {int(burst)}\n")
PYEOF
read BEST BURST < runs/n300_cfg.txt
if [ "$BEST" = "c290" ]; then POOL="--pool-c 290 --pool-k 1"; else POOL="--pool-c 240 --pool-k 32"; fi
if [ "$BURST" = "1" ]; then BOPT="--burst-len 64 --burst-every 8"; else BOPT=""; fi
echo "=== N300 CONFIG: $BEST burst=$BURST gen-refresh=$GENR $OVFGEN ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 300 --serve-mode nosync --gen-refresh $GENR ${=OVFGEN} ${=BOPT} ${=POOL} --orders $ORD --tag c80n300 >> runs/campana.log 2>&1 && echo '=== N300 OK ===' >> $LOG
echo "=== NOCHE567 COMPLETO $(date) ===" >> $LOG
