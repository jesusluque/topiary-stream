"""Shared helpers: seeds, corpus I/O, MoE block discovery, bit-plane packing.

Everything here is model-agnostic: MoE blocks are found by *content* (a module
holding a gate/router plus stacked 3-D expert tensors), never by hardcoded
names. Discovery also handles wrapped language models (`model.language_model`)
and architectures where the decoder layer *is* the MoE block (router and
experts as direct children, no container).

Bit-plane helpers implement the packed-code arithmetic the whole runtime rests
on: a 4-bit affine tensor splits exactly into two 2-bit planes
(q4 = 4*q_hi + q_lo), each a valid tensor for MLX's standard quantized kernels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

GATE_NAMES = ("gate", "router", "gate_proj", "wg")
EXPERTS_NAMES = ("switch_mlp", "experts", "mlp", "moe")
GROUP = 64


def set_seeds(seed: int) -> None:
    mx.random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------- corpus I/O


def load_corpus(path: Path, limit_tokens: int) -> list[dict[str, Any]]:
    """Read calibration rows ({"text", "n_tokens"} per line) up to a token budget."""
    rows: list[dict[str, Any]] = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
            total += row["n_tokens"]
            if total >= limit_tokens:
                break
    return rows


def token_nll(logits: mx.array, ids: mx.array) -> mx.array:
    """Teacher-forced per-token NLL. logits [1,L,V], ids [1,L] -> [L-1].
    log-softmax directo (z - logsumexp): estable con vocabularios de 248k,
    donde log(softmax+eps) pierde las colas."""
    z = logits[0, :-1].astype(mx.float32)
    lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
    return -mx.take_along_axis(lp, ids[0, 1:][:, None], axis=-1).squeeze(-1)


# ---------------------------------------------------------- MoE introspection


def _find_child(module: nn.Module, candidates: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in candidates:
        child = module.get(name) if hasattr(module, "get") else None
        if isinstance(child, nn.Module):
            return name, child
    return None, None


def _has_stacked_experts(module: nn.Module) -> bool:
    """A stacked-experts container holds at least one 3-D tensor [n_experts, out, in]."""
    for _, value in module.items():
        if isinstance(value, mx.array) and value.ndim == 3:
            return True
        if isinstance(value, nn.Module) and _has_stacked_experts(value):
            return True
    return False


def moe_block_of(layer: nn.Module) -> nn.Module | None:
    """Return the MoE block inside (or equal to) a decoder layer, else None.

    Some architectures make the layer itself the block (router + experts as
    direct children) — probe the layer before its children.
    """
    candidates = [layer] + [c for _, c in layer.items() if hasattr(c, "items")]
    for cand in candidates:
        gate_name, _ = _find_child(cand, GATE_NAMES)
        _, experts = _find_child(cand, EXPERTS_NAMES)
        if gate_name and experts is not None and _has_stacked_experts(experts):
            return cand
    return None


def find_moe_blocks(model) -> list[tuple[int, nn.Module]]:
    """All (layer_index, moe_block) pairs; unwraps `language_model` containers."""
    inner = getattr(model, "language_model", model)
    layers = getattr(inner, "layers", None) or inner.model.layers
    blocks = []
    for i, layer in enumerate(layers):
        block = moe_block_of(layer)
        if block is not None:
            blocks.append((i, block))
    return blocks


# ------------------------------------------------------------------ bit planes


def unpack4(wq: mx.array) -> np.ndarray:
    """Packed 4-bit uint32 tensor [..., words] -> codes [..., words*8] (0..15).

    MLX packs codes little-endian within each 32-bit word; viewing the raw
    bytes yields two codes per byte, low nibble first.
    """
    shape = wq.shape
    by = np.frombuffer(np.array(wq, copy=True).tobytes(), dtype=np.uint8)
    by = by.reshape(shape[:-1] + (shape[-1] * 4,))
    codes = np.empty(shape[:-1] + (shape[-1] * 8,), dtype=np.uint8)
    codes[..., 0::2] = by & 15
    codes[..., 1::2] = by >> 4
    return codes


def pack2(codes: np.ndarray) -> mx.array:
    """2-bit codes [..., n] (n % 16 == 0) -> packed uint32 [..., n/16]."""
    lead = codes.shape[:-1]
    q = codes.reshape(lead + (codes.shape[-1] // 16, 16)).astype(np.uint32)
    packed = np.zeros(lead + (codes.shape[-1] // 16,), dtype=np.uint32)
    for j in range(16):
        packed |= q[..., j] << (2 * j)
    return mx.array(packed)


def pack(codes: np.ndarray, bits: int) -> mx.array:
    """Generic little-endian packer for 2/4/8-bit codes (2-D input)."""
    per_word = 32 // bits
    q = codes.reshape(codes.shape[0], -1, per_word).astype(np.uint32)
    packed = np.zeros(q.shape[:2], dtype=np.uint32)
    for j in range(per_word):
        packed |= q[:, :, j] << (bits * j)
    return mx.array(packed)


def unpack_bits(wq: mx.array, bits: int) -> np.ndarray:
    """Packed uint32 2-D tensor -> codes, word-major little-endian order."""
    words = np.array(wq, copy=True).view(np.uint32)
    per_word = 32 // bits
    mask = (1 << bits) - 1
    cols = [(words >> (bits * j)) & mask for j in range(per_word)]
    return np.stack(cols, axis=-1).reshape(words.shape[0], -1).astype(np.uint16)


# ------------------------------------------------------------ routing families


def block_top_k(blk) -> int:
    """k of a MoE block across families (Qwen: top_k; Mixtral: num_experts_per_tok;
    DeepSeek: the gate's own top_k)."""
    for name in ("top_k", "num_experts_per_tok"):
        v = getattr(blk, name, None)
        if isinstance(v, int):
            return v
    g = getattr(blk, "gate", None)
    if g is not None and isinstance(getattr(g, "top_k", None), int):
        return g.top_k
    raise AttributeError(f"cannot determine top_k of {type(blk).__name__}")


def route(blk, x_flat: mx.array):
    """(inds [T,k], scores [T,k]) exactly as the block's own forward would
    compute them, for every family the runtime supports:

      Qwen3 / Qwen3.5 / Qwen3-Next  gate = Linear; softmax over all experts,
                                    top-k, renormalised iff norm_topk_prob
      Mixtral                       gate = Linear; softmax over the selected
                                    logits (== renormalised top-k probs)
      DeepSeek-V2                   gate = MoEGate returning (inds, scores)
                                    (its own top-k, group limits and scaling)

    Routing is never altered by the runtime: only *what is served* changes.
    """
    k = block_top_k(blk)
    out = blk.gate(x_flat)
    if isinstance(out, (tuple, list)):
        inds, scores = out
        return mx.stop_gradient(inds), scores.astype(mx.float32)
    gates = mx.softmax(out.astype(mx.float32), axis=-1, precise=True)
    inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    renorm = getattr(blk, "norm_topk_prob", None)
    if renorm is None:
        renorm = type(blk).__name__.startswith("Mixtral")
    if renorm:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


def add_shared(blk, x_flat: mx.array, y: mx.array) -> mx.array:
    """Shared-expert term per family: Qwen (sigmoid-gated `shared_expert`),
    DeepSeek (ungated `shared_experts`), Mixtral (none)."""
    if hasattr(blk, "shared_expert"):
        return y + mx.sigmoid(blk.shared_expert_gate(x_flat)) * blk.shared_expert(x_flat)
    if hasattr(blk, "shared_experts"):
        return y + blk.shared_experts(x_flat)
    return y
