#!/bin/zsh
# Upload a Topiary checkpoint to Hugging Face as a PRIVATE model repo.
# Kept private until method + models are published together (checkpoints are
# self-documenting: shrunken moe_intermediate_size + ordered salience is enough
# for an expert to reverse-engineer the technique).
#
# Usage: ./upload.sh <local_model_dir> <hf_user/repo_name>
# Requires: hf auth login (once), and a README.md inside the model dir
# (start from MODEL_CARD_TEMPLATE.md and fill the {PLACEHOLDERS}).

set -euo pipefail
MODEL_DIR=${1:?usage: upload.sh <local_model_dir> <hf_user/repo>}
HF_REPO=${2:?usage: upload.sh <local_model_dir> <hf_user/repo>}

[[ -f "$MODEL_DIR/README.md" ]] || { echo "missing $MODEL_DIR/README.md (model card)"; exit 1; }
grep -qE '\{[A-Z][A-Z0-9_]*\}' "$MODEL_DIR/README.md" && { echo "model card still has unfilled {PLACEHOLDERS}"; exit 1; }

hf repo create "$HF_REPO" --repo-type model --private || true
hf upload "$HF_REPO" "$MODEL_DIR" . --repo-type model
# ENFORCE + VERIFY privacy after upload. Lesson learned the hard way: a silently
# failing create (bad flag, swallowed stderr) let `hf upload` auto-create the
# repo PUBLIC. Never trust the create step alone.
python3 - "$HF_REPO" <<'PYEOF'
import sys
from huggingface_hub import HfApi
api = HfApi()
repo = sys.argv[1]
api.update_repo_settings(repo, private=True)
assert api.model_info(repo).private, f"{repo} is still PUBLIC"
print(f"[verified] {repo} is private")
PYEOF
echo "[done] https://huggingface.co/$HF_REPO (private, verified)"
