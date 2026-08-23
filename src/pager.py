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
OVF = 32          # filas del tier de desbordamiento (refresh barato)


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
        # Tier de DESBORDAMIENTO (refresh barato): OVF filas de P0 donde
        # entran los expertos nuevos a cadencia fina (copias de ~27 MB en
        # vez de ~200 MB por proyección); se funden en el pool grande en el
        # refresh completo (cada `ovf_merge` refreshes rápidos).
        self.lookup_o = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self.ovf_members: list[int] = [-1] * OVF
        self.ovf_next = 0
        self._install(list(np.argsort(-self.ema)[: self.c]))
        for proj in PARTS:
            m = self.mm[proj]
            self.pools[proj]["po"] = mx.zeros((OVF, m["out"], m["p0"].shape[1] // m["out"]), dtype=mx.uint32)
            self.pools[proj]["so"] = mx.zeros((OVF,) + m["s_shape"], dtype=self.pools[proj]["s0"].dtype)
            self.pools[proj]["bo"] = mx.zeros((OVF,) + m["s_shape"], dtype=self.pools[proj]["s0"].dtype)
            mx.eval(self.pools[proj]["po"], self.pools[proj]["so"], self.pools[proj]["bo"])

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

    def refresh_fast(self, counts: np.ndarray) -> int:
        """Refresh BARATO: los expertos que deberían entrar van al tier de
        desbordamiento (servidos a suelo P0) sin tocar el pool grande."""
        self.ema = 0.8 * self.ema + counts
        ideal = np.argsort(-self.ema)[: self.c]
        in0 = set(self.members0); ino = set(e for e in self.ovf_members if e >= 0)
        entering = [int(e) for e in ideal if int(e) not in in0 and int(e) not in ino][:MAX_CHURN]
        if not entering:
            return 0
        lo = np.array(self.lookup_o, copy=True)
        slots = []
        for e in entering:
            slot = self.ovf_next % OVF
            self.ovf_next += 1
            old_e = self.ovf_members[slot]
            if old_e >= 0:
                lo[old_e] = -1
            self.ovf_members[slot] = e
            lo[e] = slot
            slots.append(slot)
        self.lookup_o = mx.array(lo)
        pos_mx = mx.array(np.array(slots, dtype=np.int32))
        for proj in PARTS:
            p = self.pools[proj]
            p["po"][pos_mx] = self._rows(proj, "p0", entering)
            s_new, b_new = self._sb(proj, entering)
            p["so"][pos_mx] = s_new
            p["bo"][pos_mx] = b_new + 1.5 * s_new
            mx.eval(p["po"], p["so"], p["bo"])
        return len(entering)

    def clear_overflow(self) -> None:
        self.lookup_o = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self.ovf_members = [-1] * OVF

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
        if S.get("ema_mass"):
            S["pending"][id(self)].append((inds, scores))   # masa de gate, no cuenta
        else:
            S["pending"][id(self)].append(inds)

        pos0 = st.lookup0[inds]
        pos1 = st.lookup1[inds]
        m0 = (pos0 >= 0).astype(x_flat.dtype)[..., None, None]
        m1 = (pos1 >= 0).astype(x_flat.dtype)[..., None, None]
        r0, r1 = mx.maximum(pos0, 0), mx.maximum(pos1, 0)
        xx = mx.expand_dims(x_flat, (-2, -3))
        mode = S["mode"]

        # EXACT PREFILL for pool policies: the prompt is one batched pass —
        # serve it at true 4-bit and keep the pool policy for decode only.
        # Removes the teacher-forced/prefill toll (measured +28%/+120% on an
        # 80B) and starts decode from an exact KV.
        if mode == "exact" or (x_flat.shape[0] > 1 and mode in ("nosync", "floor2d", "absorb")):
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
        elif mode == "absorb":
            # El experto COMPARTIDO absorbe la masa de gate de los caídos:
            # un suelo universal ya residente y entrenado, a coste cero.
            sv = (pos0 >= 0).astype(scores.dtype)
            dropped = (scores * (1 - sv)).sum(axis=-1, keepdims=True)
            scores = scores * sv
            y = y.squeeze(-2)
            y = (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
            if hasattr(self, "shared_expert"):
                y = y + dropped.astype(y.dtype) * self.shared_expert(x_flat)
            return _plus_shared(self, x_flat, y).reshape(shape)
        else:  # nosync: drop-renormalize outside the pool (+ tier de desbordamiento)
            if S.get("ovf_merge", 0):
                pos_o = st.lookup_o[inds]
                m_o = ((pos_o >= 0) & (pos0 < 0)).astype(x_flat.dtype)[..., None, None]
                r_o = mx.maximum(pos_o, 0)

                def ovf_part(proj, xin):
                    p = st.pools[proj]
                    return mx.gather_qmm(xin, p["po"], p["so"] * 4, p["bo"],
                                         rhs_indices=r_o, transpose=True,
                                         group_size=GROUP, bits=2) * m_o
                h_o = glu.activation(ovf_part("up_proj", xx), ovf_part("gate_proj", xx))
                y = y + ovf_part("down_proj", h_o)
                sv = ((pos0 >= 0) | (pos_o >= 0)).astype(scores.dtype)
            else:
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


def _miss_rate(st, idx_np: np.ndarray) -> float:
    """Fracción de slots enrutados que NO estaban residentes (pool ni
    desbordamiento) — el sensor de dispersión del gobernador de marchas."""
    l0 = np.array(st.lookup0)
    lo = np.array(st.lookup_o)
    return float(np.mean((l0[idx_np] < 0) & (lo[idx_np] < 0)))


def _shift_gear(gear: str) -> None:
    """Cambio de marcha: reconstruir pools a (C, K) de la marcha. Raro por
    diseño (histéresis), así que el coste de _install es aceptable."""
    c, k = S["gear_cfg"][gear]
    for st in S["layers"].values():
        st.c, st.k = c, k
        st._install(list(np.argsort(-st.ema)[:c]))
        st.clear_overflow()
    S["gear"] = gear
    S["gear_dwell"] = 0


def refresh_all() -> None:
    merge = S.get("ovf_merge", 0)
    S["calls"] = S.get("calls", 0) + 1
    fast = merge > 0 and (S["calls"] % merge != 0)
    misses = []
    for bid, st in S["layers"].items():
        pend = S["pending"][bid]
        if not pend:
            continue
        if S.get("ema_mass"):
            idx_np = np.concatenate([np.array(p[0]).reshape(-1) for p in pend])
            w_np = np.concatenate([np.array(p[1]).reshape(-1) for p in pend]).astype(np.float64)
        else:
            idx_np = np.concatenate([np.array(p).reshape(-1) for p in pend])
            w_np = None
        S["pending"][bid] = []
        ok = (idx_np >= 0) & (idx_np < st.n_experts)
        idx_np = idx_np[ok]
        if w_np is not None:
            w_np = w_np[ok]
        if len(idx_np) == 0:
            continue
        if S.get("gear_cfg"):
            misses.append(_miss_rate(st, idx_np))
        # EMA por cuenta (default) o por MASA de gate (el daño de un miss es
        # proporcional a su gate: retener lo que importa perder)
        counts = np.bincount(idx_np, weights=w_np, minlength=st.n_experts).astype(np.float64)
        if w_np is not None:
            counts *= len(idx_np) / max(w_np.sum(), 1e-9)   # misma escala que la cuenta
        if fast:
            st.refresh_fast(counts)
        else:
            st.refresh(counts)
            if merge > 0:
                st.clear_overflow()
    # Gobernador de DOS MARCHAS: dispersión alta → marcha 2-bit amplia (C alto,
    # K≈0); dispersión baja → marcha 4-bit focal. Histéresis + permanencia.
    if S.get("gear_cfg") and misses:
        m = float(np.mean(misses))
        S["miss_rate"] = m
        S["gear_dwell"] = S.get("gear_dwell", 0) + 1
        if S["gear_dwell"] >= S.get("gear_min_dwell", 2):
            if S["gear"] == "lo" and m > S["gear_hi_thr"]:
                _shift_gear("hi")
                S["gear_events"] = S.get("gear_events", []) + [("hi", m)]
            elif S["gear"] == "hi" and m < S["gear_lo_thr"]:
                _shift_gear("lo")
                S["gear_events"] = S.get("gear_events", []) + [("lo", m)]
