#!/bin/zsh
# GENERALIDAD (orden del usuario 23/08): ¿la ley de cobertura sobrevive a un
# router con balanceo de carga? DeepSeek-Coder-V2-Lite-Instruct-4bit (64
# expertos enrutados top-6 + 2 compartidos, 27 capas, afín 4-bit g64).
# Cadena: control stock-vs-exacto → split full-memmap (disco externo) →
# orders → base exacta (PPL, KLD ref) → C ∈ {64 (suelo universal), 30 (47%),
# 15 (23%)} con K=16: KLD decode, batería hard (n=100) y wide (n=500), tok/s.
cd "$(dirname "$0")/.."
PY=${LAB:-$HOME/luc/nanite-moe}/.venv/bin/python
export PYTHONPATH=${LAB:-$HOME/luc/nanite-moe}/src
LOG=runs/ablation.log; OUT=runs/generality_dsv2.log
D=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/held_out.jsonl
CAL=${LAB:-$HOME/luc/nanite-moe}/data/calib_general_qwen3/calib.jsonl
ART=/Volumes/Untitled/dsv2lite-stream
A=artifacts/dsv2lite-stream
PROMPT="Write a Python function that merges two sorted lists into one sorted list."
until ! pgrep -f "hf download mlx-community/DeepSeek" > /dev/null; do sleep 60; done
SNAP=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-Coder-V2-Lite-Instruct-4bit/snapshots/*/ | head -1)
$PY - "$SNAP" <<'PYEOF' || { echo "=== GEN-DSV2 DESCARGA INCOMPLETA $(date) ===" >> runs/ablation.log; exit 1 }
import json, os, sys
snap = sys.argv[1]; idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
missing = [f for f in set(idx["weight_map"].values()) if not os.path.exists(os.path.join(snap, f))]
assert not missing, missing
PYEOF
echo "=== GEN-DSV2 START $(date) ===" >> $LOG
# control A: generación stock (mlx_lm) greedy
$PY -m mlx_lm generate --model "$SNAP" --prompt "$PROMPT" --max-tokens 48 --temp 0 > runs/gen_dsv2_stock.txt 2>&1 || echo "=== GEN-DSV2 STOCK FALLO ===" >> $LOG
# split a disco externo (no cabe en el interno junto a la caché HF)
$PY -u src/split.py --src "$SNAP" --out $ART --layout full-memmap >> $OUT 2>&1 && ln -sfn $ART $A && echo "=== GEN-DSV2 SPLIT OK ===" >> $LOG || { echo "=== GEN-DSV2 SPLIT FALLO ===" >> $LOG; exit 1 }
# control B: modo exacto del pager con el MISMO prompt (debe coincidir con A)
$PY -u src/serve.py --artifact $A --serve-mode exact --pool-c 8 --pool-k 2 --prompt "$PROMPT" --max-tokens 48 > runs/gen_dsv2_exact.txt 2>&1 || echo "=== GEN-DSV2 EXACT FALLO ===" >> $LOG
$PY - <<'PYEOF' >> $LOG
import re
a = open("runs/gen_dsv2_stock.txt").read(); b = open("runs/gen_dsv2_exact.txt").read()
def body(t):
    t = re.sub(r"\[(load|pool|warmup|serve)\][^\n]*\n?", "", t)
    t = re.sub(r"={5,}[^\n]*\n?", "", t); t = re.sub(r"Prompt:[^\n]*\n?|Generation:[^\n]*\n?|Peak memory[^\n]*\n?", "", t)
    return " ".join(t.split())
ba, bb = body(a), body(b)
n = 0
for x, y in zip(ba.split(), bb.split()):
    if x != y: break
    n += 1
print(f"=== GEN-DSV2 CONTROL stock-vs-exact: {n} tokens(words) identicos de {min(len(ba.split()), len(bb.split()))} ===")
PYEOF
# orders (prior del pool) a través del pager
$PY -u src/salience.py --artifact $A --data $CAL --tokens 6000 --out $ART/orders_routed.npz >> $OUT 2>&1 && echo "=== GEN-DSV2 ORDERS OK ===" >> $LOG || { echo "=== GEN-DSV2 ORDERS FALLO ===" >> $LOG; exit 1 }
ORD=$A/orders_routed.npz
# base exacta: KLD referencia (decode) + PPL general
$PY -u src/eval_stream.py --artifact $A --stage kld --serve-mode exact --kld-decode --data-general $D --chunks-general 4 --chunk-len 512 --kld-out runs/kldds_base_long.npz --tag dsbase >> $OUT 2>&1 && echo "=== GEN-DSV2 KLD-BASE OK ===" >> $LOG || echo "=== GEN-DSV2 KLD-BASE FALLO ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage ppl --serve-mode exact --data-general $D --chunks-general 8 --chunk-len 512 --tag dsbase >> $OUT 2>&1 && echo "=== GEN-DSV2 PPL-BASE OK ===" >> $LOG || echo "=== GEN-DSV2 PPL-BASE FALLO ===" >> $LOG
for C in 64 30 15; do
  $PY -u src/serve.py --artifact $A --serve-mode nosync --pool-c $C --pool-k 16 --orders $ORD --max-tokens 256 > /tmp/sds.txt 2>&1; echo "C=$C: $(grep -E 'tok/s' /tmp/sds.txt | tail -1)" >> $LOG
  $PY -u src/eval_stream.py --artifact $A --stage kld --serve-mode nosync --kld-decode --kld-refresh 256 --pool-c $C --pool-k 16 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kldds_base_long.npz --tag ds_c$C >> $OUT 2>&1 && echo "=== GEN-DSV2 KLD C=$C OK ===" >> $LOG || echo "=== GEN-DSV2 KLD C=$C FALLO ===" >> $LOG
  $PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 100 --serve-mode nosync --gen-refresh 128 --pool-c $C --pool-k 16 --orders $ORD --tag ds_c${C}_hard >> $OUT 2>&1 && echo "=== GEN-DSV2 HARD C=$C OK ===" >> $LOG || echo "=== GEN-DSV2 HARD C=$C FALLO ===" >> $LOG
  $PY -u src/eval_stream.py --artifact $A --stage bench --bench mmlu,lambada --n 500 --serve-mode nosync --gen-refresh 128 --pool-c $C --pool-k 16 --orders $ORD --tag ds_c${C}_wide >> $OUT 2>&1 && echo "=== GEN-DSV2 WIDE C=$C OK ===" >> $LOG || echo "=== GEN-DSV2 WIDE C=$C FALLO ===" >> $LOG
done
# base exacta en tareas (la verdad del modelo)
$PY -u src/eval_stream.py --artifact $A --stage bench --bench math500,mbpp --n 100 --serve-mode exact --tag ds_exact_hard >> $OUT 2>&1 && echo "=== GEN-DSV2 HARD EXACT OK ===" >> $LOG || echo "=== GEN-DSV2 HARD EXACT FALLO ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage bench --bench mmlu,lambada --n 500 --serve-mode exact --tag ds_exact_wide >> $OUT 2>&1 && echo "=== GEN-DSV2 WIDE EXACT OK ===" >> $LOG || echo "=== GEN-DSV2 WIDE EXACT FALLO ===" >> $LOG
echo "=== GEN-DSV2 COMPLETO $(date) ===" >> $LOG
