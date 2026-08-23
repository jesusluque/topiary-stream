# 4. Building artifacts

Everything starts from an `mlx-community` 4-bit checkpoint (affine
quantization, `group_size 64`, `bits 4`; the routers already come at 8 bits
via the checkpoint's own override). The environment is the repo's `.venv`
(`uv venv && uv pip install -e .`; mlx 0.32.0 / mlx-lm 0.31.3) and the
commands are run from the root of `topiary-stream`.

## 4.1 `split.py` — checkpoint → servable artifact

```bash
# P0 fits in RAM (35B, 30B): P0 in the checkpoint, P1 in memmaps
python src/split.py --src mlx-community/Qwen3.5-35B-A3B-4bit \
    --out artifacts/qwen35-stream --layout resident-p0

# P0 does not fit (80B, 235B): skeleton + P0/P1/sb in memmaps; --consume deletes
# each source shard after processing it (peak disk ≈ checkpoint + 1 shard)
python src/split.py --src mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit \
    --out artifacts/qwen80-stream --layout full-memmap --consume
```

What it does per shard: loads with `mx.load`, detects expert tensors by
name (`.switch_mlp.{gate,up,down}_proj.weight|scales|biases`), unpacks the
4-bit codes (`unpack4`), separates `codes >> 2` (P0) and `codes & 3` (P1),
repacks them to 2 bits (`pack2`) and writes them with `tofile` as
`uint32 [experts, words]`. In `full-memmap` it additionally stacks scales and
biases into `float16 [experts, 2, sb_cols]` and replaces each block with
`1×64` dummy tensors. Time: ~15 min for the 35B; the 80B ~1 h; the 235B
several hours (disk-bound).

Output: a self-contained directory with `config.json` (marked with
`stream_layout` and, in full-memmap, `stream_skeleton`), tokenizer and chat
template, `model*.safetensors`, `p1_manifest.json` or `stream_manifest.json`,
and the `.bin` files.

For a **Topiary** checkpoint (per-layer widths, taper) the runtime needs the
public `per_layer.maybe_patch` shim from the Topiary repo:
`PYTHONPATH=<topiary>/src`. `examples/build30stream.sh`
documents the full build of the 30B-Stream (split + smoke test + suite).

## 4.2 `salience.py` — pool prior and floor input

```bash
python src/salience.py --artifact artifacts/qwen80-stream \
    --data data/calib.jsonl   # jsonl with {"text", "n_tokens"} per line \
    --tokens 6000 --out artifacts/qwen80-stream/orders_routed.npz
```

It works **through the pager in exact mode**, with no checkpoint (in
full-memmap the original may have been consumed). It spies on the block's
forward: recomputes gate/up from the planes, accumulates `E[h²]` per (layer,
expert, neuron) over the corpus, and at the end multiplies by
`‖W_down[:, i]‖²` read from the planes. It writes `salience_{li}` of shape
`[E, inter]`. Cost ≈ 2× the exact forward plus one dequant pass per
(layer, expert). Consumers: `patch_pool(..., orders=)` (prior for the EMA:
`salience.sum(axis=1)`) and `floor.py`.

To read a model's **concentration curve** (which decides whether the
universal floor is viable): sort each row, accumulate, and look at what
fraction of the energy the 25%/50% prefix captures. Measured: 80B 25% → 61.6%;
235B 16.7% → 53.5% (flat); concentrated models ~94%.

## 4.3 `floor.py` — universal 2D floor

```bash
python src/floor.py --artifact artifacts/qwen80-stream \
    --orders artifacts/qwen80-stream/orders_routed.npz \
    --k-floor 128 --out artifacts/qwen80-floor128.safetensors
```

For each layer it takes the `k_floor` most salient neurons of each expert
(`k_floor` a multiple of 64), sorts the prefix in natural order (keeps the
column cut of `down` coherent), extracts from the P0 memmaps the rows
(gate/up) or the column groups (down) and the corresponding scales/biases,
and saves `L{li}.{proj}.{w,s,b}` and `L{li}.pref`. Sizes:
80B `k=128` (25% width) 5.6 GB; 235B `k=256` (16.7%) in the kit. It is
served with `--serve-mode floor2d --floor <file>`.

## 4.4 `pyramid.py` — anchored pyramid from an 8-bit master

```bash
python src/pyramid.py --stage anchored --src mlx-community/<model>-8bit --out models/m
#  → models/m-q4 (anchor), models/m-q8 (anchor+refinement), models/m-q2 (truncated anchor)
python src/pyramid.py --stage derive  --src <8bit> --bits 2 --out models/m-q2-derived   # negative control
python src/pyramid.py --stage requant --src <8bit> --bits 2 --out models/m-q2-native    # native baseline
```

It requires a **uniform** 8-bit master with no per-module overrides (it
checks). It is the tool behind the validation in §2.1; the serving artifacts
start directly from the community 4-bit (which is the anchor).

## 4.5 `protect.py` — tensor protection via HTTP ranges (not measured)

```bash
python src/protect.py --artifact artifacts/qwen80-stream --router \
    --router-repo Qwen/Qwen3-Next-80B-A3B-Instruct          # routers to official BF16
python src/protect.py --artifact artifacts/qwen80-stream --skeleton8 \
    --skel-repo mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit   # non-expert to 8 bits
```

It reads only the needed tensors from the remote safetensors (header +
`Range GET`), writes a **new artifact** (`qwen80-prot`, `qwen80-prot8`)
with the `.bin` files/manifests symlinked and per-path overrides in
`config.quantization` (`False` = unquantized; `{"bits": 8}`). It never
modifies the production artifact in place. Cost: +1.15 GB for the 8-bit
skeleton. It was left **unmeasured** (campaign cancelled); correction on
record: the router was already at 8 bits, so `--router` is 8→16,
expectation ≈ 0.

## 4.6 `awq_master.py` — salience-protected master (parked)

It registers an `AWQConfig` for `qwen3_moe` (llama-style attention + scales
for `switch_mlp` down/gate/up) in `mlx_lm.quant.awq` and delegates to its
`main`. It is the only untested lever for the quality of the 2-bit plane
itself (§6.6); parked for disk reasons (it needs the BF16 master) and by the
"no clear return, no" rule.

## 4.7 Artifact inventory (2026-08-23)

| Artifact | Layout | Size | Origin | Notes |
|---|---|---|---|---|
| `artifacts/qwen30-stream` | resident-p0 | 13 GB | `qwen3-30b-topiary` (taper, own checkpoint) | champion at 9.2 GB; channels ordered by salience |
| `artifacts/qwen35-stream` | resident-p0 | — | `mlx-community/Qwen3.5-35B-A3B-4bit` | rebuilt from the community checkpoint with `split.py`; published on HF |
| `artifacts/qwen80-stream` | full-memmap | 42 GB | `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` | + `orders_routed.npz`; 96 overrides in config (8-bit routers) |
| `artifacts/qwen80-floor128.safetensors` | 2D floor | 5.6 GB | floor.py k=128 | does not fit alongside C=240 |
| `artifacts/qwen80-prot`, `-prot8` | full-memmap (symlinks) | 1.3 / 2.4 GB | protect.py | unmeasured |
| `artifacts/qwen235-stream` | full-memmap | 134 GB | `mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit-DWQ` (DWQ shards mixed with plain 4-bit siblings) | complete |
| `artifacts/qwen235-stream-kit` | skeleton + floor256 + orders | 15 GB | — | what gets uploaded to HF: the user regenerates the planes with `split.py` |
| `artifacts/unsloth/…UD-Q2_K_XL.gguf` | GGUF (the rival) | 28 GB | `unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF` | only runs with `-ngl 0` on 24 GB |

Rule: the runtime never deletes anything except under an explicit
`--consume`.

## 4.8 Publishing on Hugging Face

Repos (private as of today, pending the flip):
`jesusluque/qwen3.5-35b-topiary-stream`, `qwen3-next-80b-topiary-stream`,
`qwen3-235b-topiary-stream-kit`. The `test_model_card_placeholder_gate` test
prevents uploading cards with unfilled markers. The Topiary checkpoints
(`qwen3-30b-topiary`, `-w640`, `-w576-code`) are public and their cards
already carry the note on the price of the taper (−10 MMLU).
