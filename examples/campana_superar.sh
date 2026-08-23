#!/bin/zsh
# CAMPAÑA "SUPERAR LA TABLA": (1) KLD de candidatos; (2) elegir el mejor con
# velocidad >= rival (12.6 tok/s); (3) batería completa con esa config.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
until grep -q "RIVAL-KLD COMPLETO\|RIVAL-KLD SERVER MUERTO\|RIVAL-KLD HUMO FALLO" runs/ablation.log 2>/dev/null && ! pgrep -f "eval_stream|llama-" > /dev/null; do sleep 30; done
echo "=== CAMPANA START $(date) ===" >> $LOG
kld() {  # nombre modo refresh
  $PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --kld-refresh $3 --serve-mode $2 --pool-c 240 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag cand_$1 >> runs/kld80.log 2>&1 && echo "=== CAND $1 OK ===" >> $LOG
}
kld r128 nosync 128
kld r64abs absorb 64
kld r32 nosync 32
# velocidad de cada candidato (256 tok)
for c in "r128 nosync 128" "r64 nosync 64" "r64abs absorb 64" "r32 nosync 32"; do
  set -- ${=c}
  $PY -u src/serve.py --artifact $A --pool-c 240 --pool-k 32 --orders $ORD --serve-mode $2 --refresh $3 --max-tokens 256 2>&1 | grep -E "tok/s" | sed "s/^/[speed $1] /" >> runs/campana.log
done
# elegir: menor KLD con tok/s >= 12.6
$PY - <<'PYEOF' >> runs/campana.log 2>&1
import json, re
cands = {"r64": ("nosync", 64), "r128": ("nosync", 128), "r64abs": ("absorb", 64), "r32": ("nosync", 32)}
kl = {}
for c in cands:
    tag = "k80c240_r64" if c == "r64" else f"cand_{c}"
    try: kl[c] = json.load(open(f"runs/kld_{tag}.json"))["kld_mean"]
    except Exception: pass
sp = {}
for line in open("runs/campana.log"):
    m = re.search(r"\[speed (\w+)\].*= ([\d.]+) tok/s", line)
    if m: sp[m.group(1)] = float(m.group(2))
ok = {c: kl[c] for c in kl if sp.get(c, 0) >= 12.6}
best = min(ok, key=ok.get) if ok else min(kl, key=kl.get)
print("KLD:", kl, "| tok/s:", sp, "| ELEGIDO:", best)
open("runs/campana_best.txt", "w").write(f"{cands[best][0]} {cands[best][1]}\n")
PYEOF
read MODE REF < runs/campana_best.txt
echo "=== CAMPANA CONFIG: mode=$MODE refresh=$REF ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 100 --serve-mode $MODE --gen-refresh $REF --pool-c 240 --pool-k 32 --orders $ORD --tag c80best_hard >> runs/campana.log 2>&1 && echo '=== CAMPANA HARD OK ===' >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench mmlu --n 500 --serve-mode $MODE --gen-refresh $REF --pool-c 240 --pool-k 32 --orders $ORD --tag c80best_wide >> runs/campana.log 2>&1 && echo '=== CAMPANA WIDE OK ===' >> $LOG
echo "=== CAMPANA COMPLETO $(date) ===" >> $LOG
