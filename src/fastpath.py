"""Fast-path runtime for `resident-p0` artifacts: sync-free decode + governor.

Design (each point bought with a measurement):

  1. Pool-MEMBERSHIP policy, not per-token thresholds: K experts per layer
     keep their P1 plane on-GPU. Membership is encoded in a dynamic *biases*
     tensor — members carry beta (their P1 completes them to exact 4-bit),
     non-members carry beta+1.5s (the centroid floor) — so ONE gather_qmm
     serves hot and cold. This removed the 40 CPU-GPU syncs/token that held
     the reference 35B at 8.1 tok/s; the fast path sustains 44.5.
  2. A miss NEVER blocks: an absent plane serves the P0 floor this token
     ("blurry for a frame") and the pool learns it at the next refresh.
  3. ONE deferred sync per REFRESH tokens: routing counts feed an EMA
     (frequency + recency — the signal that predicts MoE locality), pages
     come from np.memmap (only touched rows are read).
  4. The governor makes residency elastic: a vm_stat pressure loop resizes K
     live. Measured: four automatic shrink steps under an 8 GB external
     balloon with generation completing cleanly.

Usage (via serve.py):
    python src/serve.py --artifact artifacts/qwen35-stream --pool-k 32 --governor
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import GROUP, find_moe_blocks

PARTS = ("gate_proj", "up_proj", "down_proj")


class FastLayer:
    """Per-layer state: resident P0 + on-GPU P1 pool [K] + dynamic biases."""

    def __init__(self, blk, art_dir: Path, by_proj: dict, pool_k: int):
        self.pool_k = pool_k
        self.projs = {}
        glu = blk.switch_mlp
        for name in PARTS:
            proj = getattr(glu, name)
            meta = by_proj[name]
            mm = np.memmap(art_dir / f"L{meta['layer']}.{name}.p1.bin",
                           dtype=np.uint32, mode="r",
                           shape=(meta["experts"], meta["words"]))
            e = meta["experts"]
            s, b = proj.scales, proj.biases
            b_dyn = b + 1.5 * s   # everyone starts cold (centroid floor)
            mx.eval(b_dyn)
            self.projs[name] = {
                "p0": proj.weight, "s": s, "b": b, "mm": mm,
                "out": meta["out"], "b_dyn": b_dyn,
                "pool": mx.zeros((pool_k, meta["out"],
                                  meta["words"] // meta["out"]), dtype=mx.uint32),
                "pool_s": mx.zeros((pool_k,) + tuple(s.shape[1:]), dtype=s.dtype),
            }
        self.n_experts = e
        self.lookup = mx.full((e,), -1, dtype=mx.int32)
        self.members: list[int] = []
        self.ema = np.zeros(e, dtype=np.float64)

    def refresh(self, counts: np.ndarray) -> int:
        """EMA + recency -> top-K; page new members from the memmap."""
        self.ema = 0.7 * self.ema + counts
        want = list(np.argsort(-self.ema)[: self.pool_k])
        entered = [e for e in want if e not in self.members]
        if not entered and self.members:
            return 0
        self.members = want
        lookup = np.full(self.n_experts, -1, dtype=np.int32)
        lookup[want] = np.arange(len(want))
        self.lookup = mx.array(lookup)
        want_mx = mx.array(np.array(want, dtype=np.int32))
        for name in PARTS:
            p = self.projs[name]
            rows = np.asarray(p["mm"][want])          # only pool pages read
            p["pool"] = mx.array(rows).reshape(len(want), p["out"], -1)
            p["pool_s"] = p["s"][want_mx]
            # dynamic biases: members beta (exact with P1), rest beta+1.5s
            # (all in mx — numpy has no bf16)
            b_dyn = p["b"] + 1.5 * p["s"]
            b_dyn[want_mx] = p["b"][want_mx]
            p["b_dyn"] = b_dyn
            mx.eval(p["pool"], p["pool_s"], p["b_dyn"], self.lookup)
        return len(entered)


STATE: dict = {}


def patch_fast(model, art_dir: Path, pool_k: int) -> None:
    from mlx_lm.models import switch_layers as sl

    manifest = json.load(open(art_dir / "p1_manifest.json"))
    by_layer: dict[int, dict] = {}
    for _, meta in manifest.items():
        by_layer.setdefault(meta["layer"], {})[meta["proj"]] = meta

    blocks = find_moe_blocks(model)
    block_cls = type(blocks[0][1])
    layers = {id(b): FastLayer(b, art_dir, by_layer[li], pool_k)
              for li, b in blocks}
    STATE.update({"layers": layers, "pending": {id(b): [] for _, b in blocks}})

    def patched(self, x):
        st = STATE["layers"][id(self)]
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1]) if x.ndim > 2 else x
        gates = mx.softmax(self.gate(x_flat).astype(mx.float32), axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(self, "norm_topk_prob", False):
            scores = scores / scores.sum(axis=-1, keepdims=True)
        if x_flat.shape[0] > 1:
            mx.eval(inds)   # prefill: pin now — lazy inds under memory
                            # pressure have produced garbage indices
        STATE["pending"][id(self)].append(inds)

        glu = self.switch_mlp
        xx = mx.expand_dims(x_flat, (-2, -3))
        do_sort = inds.size >= 64
        idx = inds
        inv_order = None
        if do_sort:
            xx, idx, inv_order = sl._gather_sort(xx, inds)
        member_pos = st.lookup[idx]
        in_pool = (member_pos >= 0)
        remap = mx.maximum(member_pos, 0)
        mask = in_pool.astype(x_flat.dtype)[..., None, None]

        def pf(name, xin):
            p = st.projs[name]
            y = mx.gather_qmm(xin, p["p0"], p["s"] * 4, p["b_dyn"],
                              rhs_indices=idx, transpose=True,
                              group_size=GROUP, bits=2, sorted_indices=do_sort)
            y1 = mx.gather_qmm(xin, p["pool"], p["pool_s"],
                               mx.zeros_like(p["pool_s"]), rhs_indices=remap,
                               transpose=True, group_size=GROUP, bits=2)
            return y + y1 * mask

        h = glu.activation(pf("up_proj", xx), pf("gate_proj", xx))
        y = pf("down_proj", h)
        if do_sort:
            y = sl._scatter_unsort(y, inv_order, inds.shape)
        y = y.squeeze(-2)
        y = (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
        if hasattr(self, "shared_expert"):
            y = y + mx.sigmoid(self.shared_expert_gate(x_flat)) * self.shared_expert(x_flat)
        return y.reshape(shape)

    block_cls.__call__ = patched


def refresh_all() -> int:
    """The one deferred sync: drain routing counts, refresh every pool."""
    entered = 0
    for bid, st in STATE["layers"].items():
        pend = STATE["pending"][bid]
        if not pend:
            continue
        idx_np = np.concatenate([np.array(p).reshape(-1) for p in pend])
        STATE["pending"][bid] = []
        idx_np = idx_np[(idx_np >= 0) & (idx_np < st.n_experts)]  # hardening
        if len(idx_np) == 0:
            continue
        entered += st.refresh(np.bincount(idx_np, minlength=st.n_experts)
                              .astype(np.float64))
    return entered


# -------------------------------------------------------------- the governor


def available_gb() -> float:
    """Real macOS available RAM (free+inactive+purgeable+speculative)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = {}
    for line in out.splitlines():
        if ":" in line:
            kk, v = line.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                pages[kk.strip()] = int(v)
    n = (pages.get("Pages free", 0) + pages.get("Pages inactive", 0)
         + pages.get("Pages purgeable", 0) + pages.get("Pages speculative", 0))
    return n * 16384 / 1e9


def govern(low: float = 4.0, high: float = 7.0,
           k_min: int = 4, k_max: int = 48, step: int = 8) -> str | None:
    """Elastic residency: pressure shrinks K, headroom grows it. The next
    refresh materializes the change (pools rebuild at pool_k by design)."""
    avail = available_gb()
    layers = list(STATE["layers"].values())
    k = layers[0].pool_k
    new_k = k
    if avail < low:
        new_k = max(k_min, k - step)
    elif avail > high:
        new_k = min(k_max, k + step)
    if new_k == k:
        return None
    for st in layers:
        st.pool_k = new_k
        st.members = []           # force rebuild at the new size
    return f"[gov] avail {avail:.1f} GB -> K {k}->{new_k}"
