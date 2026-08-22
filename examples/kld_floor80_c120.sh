#!/bin/zsh
# floor2d a C=240 no cabe (pool 16.5GB + suelo 6GB → thrashing, 0.1 tok/s).
# Config sana: floor2d con C=120 (10GB + 6GB). Pregunta doble: ¿el suelo
# universal rescata al C=120 (que rompió MATH -20 sin suelo)? y ¿cuánto
# baja la KLD vs los 0.774 del drop a C=240? Solo pasada servida.
cd "$(dirname "$0")/.."
PY=/Users/muriel/luc/nanite-moe/.venv/bin/python
export PYTHONPATH=/Users/muriel/luc/nanite-moe/src
LOG=runs/ablation.log
A=artifacts/qwen80-stream; ORD=artifacts/qwen80-stream/orders_routed.npz
F=artifacts/qwen80-floor128.safetensors
D=/Users/muriel/luc/nanite-moe/data/calib_general_qwen3/held_out.jsonl
echo "=== KLD-FLOOR-C120 START $(date) ===" >> $LOG
$PY -u src/eval_stream.py --artifact $A --stage kld --kld-decode --serve-mode floor2d --floor $F --pool-c 120 --pool-k 32 --orders $ORD --data-general $D --chunks-general 4 --chunk-len 512 --kld-ref runs/kld80_base_long.npz --tag k80floor120_long >> runs/kld80.log 2>&1 && echo '=== KLD-FLOOR-C120 OK ===' >> $LOG
echo "=== KLD-FLOOR COMPLETO $(date) ===" >> $LOG
