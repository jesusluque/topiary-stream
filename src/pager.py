"""Unified-pool runtime for `full-memmap` artifacts (models beyond P0-in-RAM).

The checkpoint carries no experts (a 4-5 GB skeleton loads with tiny dummy
switch tensors via a constructor patch); every expert lives in row-pageable
memmaps. A per-layer pool caches C experts at P0 (servable floor) and K of
them with P1 (exact 4-bit), governed by an EMA seeded from a traffic prior
(routed-salience orders) or uniform.

Serving modes (all measured; pick per use):

  nosync    pool + drop-renormalize outside it. Fastest (21.5 tok/s on an
            80B at 17 GB); the drop toll grows with domain dispersion —
            fine for task-style decode, heavy on broad-knowledge prefill.
  exact     every slot served P0+P1 straight from the memmaps (== true
            4-bit). Batch/teacher-forced only; ~4 GB peak. This is how you
            measure a model's true base without fitting it.
  floor     blocking floor: absent experts fetched synchronously so nothing
            is ever dropped. Correct but sync-bound (~0.2 tok/s at 94
            layers) — a demonstration mode, not a serving mode.
  floor2d   universal resident floor (see floor.py) + pool. No drops, no
            syncs; floor quality depends on salience concentration.

Usage (via serve.py):
    python src/serve.py --artifact artifacts/qwen80-stream --pool-c 240 --pool-k 32
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import GROUP, find_moe_blocks

PARTS = ("gate_proj", "up_proj", "down_proj")
MAX_CHURN = 8


class PoolLayer:
    """Per-layer state: unified pool (C at P0, K of them with P1) + memmaps."""

    def __init__(self, li: int, art_dir: Path, manifest: dict,
                 prior: np.ndarray, pool_c: int, pool_k: int):
        self.c, self.k = pool_c, pool_k
        self.mm = {}
        for proj in PARTS:
            m = manifest[f"L{li}.{proj}"]
            self.mm[proj] = {
                "p0": np.memmap(art_dir / f"L{li}.{proj}.p0.bin", dtype=np.uint32,
                                mode="r", shape=(m["experts"], m["words"])),
                "p1": np.memmap(art_dir / f"L{li}.{proj}.p1.bin", dtype=np.uint32,
                                mode="r", shape=(m["experts"], m["words"])),
                "sb": np.memmap(art_dir / f"L{li}.{proj}.sb.bin", dtype=np.float16,
                                mode="r", shape=(m["experts"], 2, m["sb_cols"])),
                "out": m["out"], "s_shape": tuple(m["s_shape"]),
            }
        self.n_experts = m["experts"]
        self.ema = prior.astype(np.float64).copy()
        self.members0: list[int] = []
        self.members1: list[int] = []
        self.pools = {p: {} for p in PARTS}
        self.lookup0 = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self.lookup1 = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self._install(list(np.argsort(-self.ema)[: self.c]))

    def _rows(self, proj: str, plane: str, ids: list[int]) -> mx.array:
        m = self.mm[proj]
        rows = np.asarray(m[plane][ids])
        return mx.array(rows).reshape(len(ids), m["out"], -1)

    def _sb(self, proj: str, ids: list[int]):
        m = self.mm[proj]
        sb = np.asarray(m["sb"][ids])
        s = mx.array(sb[:, 0]).reshape((len(ids),) + m["s_shape"])
        b = mx.array(sb[:, 1]).reshape((len(ids),) + m["s_shape"])
        return s, b

    def _install(self, want0: list[int]) -> None:
        """Full rebuild — init only; refresh() is incremental (a full rebuild
        per refresh cost 10x in decode: ~13 GB of copies every 64 tokens)."""
        self.members0 = [int(e) for e in want0]
        in0 = set(self.members0)
        want1 = [int(e) for e in np.argsort(-self.ema) if int(e) in in0][: self.k]
        self.members1 = want1
        l0 = np.full(self.n_experts, -1, dtype=np.int32)
        l0[self.members0] = np.arange(len(self.members0))
        l1 = np.full(self.n_experts, -1, dtype=np.int32)
        l1[want1] = np.arange(len(want1))
        self.lookup0, self.lookup1 = mx.array(l0), mx.array(l1)
        has_p1 = np.zeros(self.n_experts, bool)
        has_p1[want1] = True
        for proj in PARTS:
            s, b = self._sb(proj, self.members0)
            hot_rows = mx.array(has_p1[self.members0].astype(np.float32))[:, None, None]
            b_dyn = b + 1.5 * s * (1 - hot_rows)
            self.pools[proj] = {
                "p0": self._rows(proj, "p0", self.members0), "s0": s, "b_dyn": b_dyn,
                "p1": self._rows(proj, "p1", self.members1),
                "s1": self._sb(proj, self.members1)[0],
            }
            mx.eval(*self.pools[proj].values())

    def refresh(self, counts: np.ndarray) -> None:
        """Incremental: only entering rows (<= MAX_CHURN) and the P1 rotation
        diff are touched (mx setitem copies whole buffers — keep it small)."""
        self.ema = 0.8 * self.ema + counts
        ideal_order = np.argsort(-self.ema)
        ideal_set = set(int(e) for e in ideal_order[: self.c])
        entering = [int(e) for e in ideal_order
                    if int(e) in ideal_set and int(e) not in set(self.members0)][:MAX_CHURN]
        if entering:
            worst = sorted(range(len(self.members0)),
                           key=lambda p: self.ema[self.members0[p]])[: len(entering)]
            l0 = np.array(self.lookup0, copy=True)
            for pos, e_new in zip(worst, entering):
                l0[self.members0[pos]] = -1
                l0[e_new] = pos
                self.members0[pos] = e_new
            self.lookup0 = mx.array(l0)
            pos_mx = mx.array(np.array(worst, dtype=np.int32))
            has_p1 = set(self.members1)
            for proj in PARTS:
                p = self.pools[proj]
                p["p0"][pos_mx] = self._rows(proj, "p0", entering)
                s_new, b_new = self._sb(proj, entering)
                p["s0"][pos_mx] = s_new
                hot = mx.array(np.array([e in has_p1 for e in entering],
                                        dtype=np.float32))[:, None, None]
                p["b_dyn"][pos_mx] = b_new + 1.5 * s_new * (1 - hot)
                mx.eval(p["p0"], p["s0"], p["b_dyn"])
        in0 = set(self.members0)
        want1 = [int(e) for e in ideal_order if int(e) in in0][: self.k]
        if want1 != self.members1:
            changed = set(self.members1) ^ set(want1)
            self.members1 = want1
            l1 = np.full(self.n_experts, -1, dtype=np.int32)
            l1[want1] = np.arange(len(want1))
            self.lookup1 = mx.array(l1)
            has_p1 = set(want1)
            pos_map = {e: p for p, e in enumerate(self.members0)}
            touch = [pos_map[e] for e in changed if e in pos_map]
            for proj in PARTS:
                p = self.pools[proj]
                p["p1"] = self._rows(proj, "p1", want1)
                p["s1"] = self._sb(proj, want1)[0]
                if touch:
                    exps = [self.members0[t] for t in touch]
                    s_t, b_t = self._sb(proj, exps)
                    hot = mx.array(np.array([e in has_p1 for e in exps],
                                            dtype=np.float32))[:, None, None]
                    p["b_dyn"][mx.array(np.array(touch, dtype=np.int32))] = (
                        b_t + 1.5 * s_t * (1 - hot))
                mx.eval(p["p1"], p["s1"], p["b_dyn"])


S: dict = {}
FLOOR: dict = {}


def load_floor(path: str, blocks) -> None:
    """Load the universal 2D floor (floor.py output); fold centroid once."""
    t = mx.load(path)
    for li, _ in blocks:
        FLOOR[li] = {}
        for proj in PARTS:
            w = t[f"L{li}.{proj}.w"]
            s = t[f"L{li}.{proj}.s"]
            b = t[f"L{li}.{proj}.b"] + 1.5 * s
            mx.eval(w, s, b)
            FLOOR[li][proj] = (w, s, b)


def maybe_patch_skeleton(model_path: str) -> bool:
    """Patch block constructors to tiny switch tensors for skeleton loads."""
    p = Path(model_path) / "config.json"
    if not p.exists() or not json.load(open(p)).get("stream_skeleton"):
        return False
    from mlx_lm.models.switch_layers import SwitchGLU

    def make(cls):
        orig = cls.__init__

        def patched(self, args):
            orig(self, args)
            self.switch_mlp = SwitchGLU(args.hidden_size, 64, 1)

        cls.__init__ = patched

    import mlx_lm.models.qwen3_moe as qm

    make(qm.Qwen3MoeSparseMoeBlock)
    try:
        import mlx_lm.models.qwen3_next as qn

        make(qn.Qwen3NextSparseMoeBlock)
    except ImportError:
        pass
    return True


def load_model(artifact: str):
    from mlx_lm import load

    if maybe_patch_skeleton(artifact):
        model, tokenizer = load(artifact)
        mx.eval(model.parameters())
        return model, tokenizer
    model, tokenizer = load(artifact, lazy=True)
    from mlx.utils import tree_flatten

    skip = tuple(f".switch_mlp.{p}" for p in PARTS)
    to_eval = [v for kk, v in tree_flatten(model.parameters())
               if not any(s in kk for s in skip)]
    mx.eval(*to_eval)
    return model, tokenizer


def patch_pool(model, art_dir: Path, pool_c: int, pool_k: int,
               orders: str | None = None) -> None:
    manifest = json.load(open(art_dir / "stream_manifest.json"))
    ordz = np.load(orders) if orders and Path(orders).exists() else None

    blocks = find_moe_blocks(model)
    block_cls = type(blocks[0][1])
    layers = {}
    for li, blk in blocks:
        n_exp = manifest[f"L{li}.gate_proj"]["experts"]
        prior = (ordz[f"salience_{li}"].sum(axis=1) if ordz is not None
                 else np.ones(n_exp))
        layers[id(blk)] = PoolLayer(li, art_dir, manifest, prior, pool_c, pool_k)
    S.update({"layers": layers, "pending": {id(b): [] for _, b in blocks},
              "layer_of": {id(b): li for li, b in blocks},
              "mode": "nosync"})

    def patched(self, x):
        st = S["layers"][id(self)]
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1]) if x.ndim > 2 else x
        gates = mx.softmax(self.gate(x_flat).astype(mx.float32), axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(self, "norm_topk_prob", False):
            scores = scores / scores.sum(axis=-1, keepdims=True)
        if x_flat.shape[0] > 1:
            mx.eval(inds)          # prefill pin (garbage-index hardening)
        S["pending"][id(self)].append(inds)

        pos0 = st.lookup0[inds]
        pos1 = st.lookup1[inds]
        m0 = (pos0 >= 0).astype(x_flat.dtype)[..., None, None]
        m1 = (pos1 >= 0).astype(x_flat.dtype)[..., None, None]
        r0, r1 = mx.maximum(pos0, 0), mx.maximum(pos1, 0)
        xx = mx.expand_dims(x_flat, (-2, -3))
        mode = S["mode"]

        if mode == "exact":
            inds_np = np.array(inds)
            uniq, inv = np.unique(inds_np, return_inverse=True)
            rme = mx.array(inv.reshape(inds_np.shape).astype(np.int32))

            def pfx(proj, xin):
                r0e = st._rows(proj, "p0", list(uniq))
                r1e = st._rows(proj, "p1", list(uniq))
                s_e, b_e = st._sb(proj, list(uniq))
                return (mx.gather_qmm(xin, r0e, s_e * 4, b_e, rhs_indices=rme,
                                      transpose=True, group_size=GROUP, bits=2)
                        + mx.gather_qmm(xin, r1e, s_e, mx.zeros_like(s_e),
                                        rhs_indices=rme, transpose=True,
                                        group_size=GROUP, bits=2))

            glu = self.switch_mlp
            h = glu.activation(pfx("up_proj", xx), pfx("gate_proj", xx))
            y = pfx("down_proj", h).squeeze(-2)
            y = (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
            return _plus_shared(self, x_flat, y).reshape(shape)

        def pool_part(proj, xin):
            p = st.pools[proj]
            y = mx.gather_qmm(xin, p["p0"], p["s0"] * 4, p["b_dyn"],
                              rhs_indices=r0, transpose=True,
                              group_size=GROUP, bits=2) * m0
            y1 = mx.gather_qmm(xin, p["p1"], p["s1"], mx.zeros_like(p["s1"]),
                               rhs_indices=r1, transpose=True,
                               group_size=GROUP, bits=2) * m1
            return y + y1

        glu = self.switch_mlp
        h = glu.activation(pool_part("up_proj", xx), pool_part("gate_proj", xx))
        y = pool_part("down_proj", h)

        if mode == "floor2d":
            fl = FLOOR[S["layer_of"][id(self)]]
            m_f = (pos0 < 0).astype(x_flat.dtype)[..., None, None]

            def floor_part(proj, xin):
                w, s, b = fl[proj]
                return mx.gather_qmm(xin, w, s * 4, b, rhs_indices=inds,
                                     transpose=True, group_size=GROUP, bits=2)

            h_f = glu.activation(floor_part("up_proj", xx), floor_part("gate_proj", xx))
            y = y + floor_part("down_proj", h_f) * m_f
        elif mode == "floor":
            miss_np = np.array(pos0 < 0)
            if miss_np.any():
                inds_np = np.array(inds)
                miss_ids = list(np.unique(inds_np[miss_np]))
                remap_miss = np.zeros(st.n_experts, dtype=np.int32)
                remap_miss[miss_ids] = np.arange(len(miss_ids))
                rm = mx.array(remap_miss)[inds]
                mm_mask = mx.array(miss_np.astype(np.float32))[..., None, None]

                def floor_fetch(proj, xin):
                    rows = st._rows(proj, "p0", miss_ids)
                    s_m, b_m = st._sb(proj, miss_ids)
                    return mx.gather_qmm(xin, rows, s_m * 4, b_m + 1.5 * s_m,
                                         rhs_indices=rm, transpose=True,
                                         group_size=GROUP, bits=2)

                h_m = glu.activation(floor_fetch("up_proj", xx),
                                     floor_fetch("gate_proj", xx))
                y = y + floor_fetch("down_proj", h_m) * mm_mask
        else:  # nosync: drop-renormalize outside the pool
            sv = (pos0 >= 0).astype(scores.dtype)
            scores = scores * sv
            scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-9)

        y = y.squeeze(-2)
        y = (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
        return _plus_shared(self, x_flat, y).reshape(shape)

    block_cls.__call__ = patched


def _plus_shared(blk, x_flat, y):
    if hasattr(blk, "shared_expert"):
        y = y + mx.sigmoid(blk.shared_expert_gate(x_flat)) * blk.shared_expert(x_flat)
    return y


def refresh_all() -> None:
    for bid, st in S["layers"].items():
        pend = S["pending"][bid]
        if not pend:
            continue
        idx_np = np.concatenate([np.array(p).reshape(-1) for p in pend])
        S["pending"][bid] = []
        idx_np = idx_np[(idx_np >= 0) & (idx_np < st.n_experts)]
        if len(idx_np) == 0:
            continue
        st.refresh(np.bincount(idx_np, minlength=st.n_experts).astype(np.float64))
