#!/bin/zsh
# Citable throughput benchmark — run this AFTER a fresh reboot (and after
# waiting ~10 min for Spotlight/mds to settle; check `vm_stat` shows low
# pressure). Discipline: INTERLEAVED rounds — consecutive per-config repeats
# confound policy with memory residency and thermal state.
#
# Usage: ./examples/citable_bench.sh 2>&1 | tee runs/citable_$(date +%Y%m%d).log
set -euo pipefail
cd "$(dirname "$0")/.."

A35=artifacts/qwen35-stream
A80=artifacts/qwen80-stream
ORD80=artifacts/qwen80-stream/orders_routed.npz

echo "=== citable bench $(date) · uptime: $(uptime) ==="
sysctl vm.swapusage

for round in 1 2 3; do
  echo "--- round $round: 35B ---"
  python src/serve.py --artifact $A35 --pool-k 32 --max-tokens 1024 \
      --prompt "Explain how a B-tree differs from an LSM tree, with examples." \
      | tail -1
  echo "--- round $round: 80B ---"
  python src/serve.py --artifact $A80 --pool-c 240 --pool-k 32 --orders $ORD80 \
      --max-tokens 1024 \
      --prompt "Explain how a B-tree differs from an LSM tree, with examples." \
      | tail -1
done
sysctl vm.swapusage
echo "=== done — median of 3 interleaved rounds is the citable figure ==="
