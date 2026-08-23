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
import re
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import GROUP, add_shared, find_moe_blocks, route

PARTS = ("gate_proj", "up_proj", "down_proj")


class FastLayer:
    """Per-layer state: resident P0 + on-GPU P1 pool [K] + dynamic biases."""

    def __init__(self, blk, art_dir: Path, by_proj: dict, pool_k: int,
                 centroid: str = "uniform", p1_frac: str | float = 1.0):
        self.pool_k = pool_k
        # fracción del prefijo P1 por proyección: "0.5" uniforme o "g,u,d"
        parts = str(p1_frac).split(",")
        fracs = dict(zip(PARTS, map(float, parts * 3 if len(parts) == 1 else parts)))
        self.p1_frac = fracs
        self.projs = {}
        glu = blk.switch_mlp
        for name in PARTS:
            proj = getattr(glu, name)
            meta = by_proj[name]
            mm = np.memmap(art_dir / f"L{meta['layer']}.{name}.p1.bin",
                           dtype=np.uint32, mode="r",
                           shape=(meta["experts"], meta["words"]))
            s, b = proj.scales, proj.biases
            if centroid == "empirical":
                # suelo informado: el bias frío pliega la MEDIA REAL de los
                # 2 bits descartados por grupo (no el 1.5 uniforme). Mismo
                # formato, mismos bytes; solo cambia la constante del pliegue.
                qbar = _p1_group_means(mm, meta["out"],
                                       art_dir / f"L{meta['layer']}.{name}.cbias.npy")
                b_cold = b + s * mx.array(qbar)
            else:
                b_cold = b + 1.5 * s
            b_dyn = b_cold   # everyone starts cold (centroid floor)
            mx.eval(b_dyn)
            # Prefijo-P1 (4:2:2 sobre la dimensión intermedia): los canales de
            # los checkpoints topiary vienen ORDENADOS por saliencia, así que
            # "los R primeros" = "los R más salientes". gate/up recortan filas
            # de salida (R); down recorta grupos de entrada (Rg = R/64).
            in_words = meta["words"] // meta["out"]
            frac = fracs[name]
            if frac == 0:           # sin P1: la proyección vive solo del suelo
                r, pool_shape, s_pool_shape = 0, (0,), (0,)
            elif name == "down_proj":
                rg = max(1, round(frac * in_words // 4))
                pool_shape = (pool_k, meta["out"], rg * 4)
                s_pool_shape = (pool_k, s.shape[1], rg)
                r = rg * 4          # words por fila que se cargan del memmap
            else:
                r = max(64, int(round(frac * meta["out"] / 64)) * 64)
                r = min(r, meta["out"])
                pool_shape = (pool_k, r, in_words)
                s_pool_shape = (pool_k, r, s.shape[2])
            self.projs[name] = {
                "p0": proj.weight, "s": s, "b": b, "b_cold": b_cold, "mm": mm,
                "out": meta["out"], "b_dyn": b_dyn, "r": r,
                "pool": mx.zeros(pool_shape, dtype=mx.uint32),
                "pool_s": mx.zeros(s_pool_shape, dtype=s.dtype),
            }
        n = by_proj["gate_proj"]["experts"]
        self.n_experts = n
        self.lookup = mx.full((n,), -1, dtype=mx.int32)
        self.members: list[int] = []
        self.ema = np.zeros(n, dtype=np.float64)

    def refresh(self, counts: np.ndarray) -> int:
        """EMA + recency -> top-K; page new members from the memmap.

        Nota deliberada: aquí el pool se repagina ENTERO ante cualquier
        cambio (a diferencia del refresh incremental del pager). Es
        aceptable porque las filas del resident-p0 son pequeñas: K=32
        completos son ~40 MB desde page cache, una vez cada REFRESH tokens
        — medido: refresh=64 cuesta solo −2.5% tok/s vs refresh=256. En el
        pager (80B, C=240) las filas son ~10× mayores y la reconstrucción
        completa costaba 10×: allí lo incremental es obligatorio."""
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
            if p["r"] == 0:
                continue
            rows = np.asarray(p["mm"][want])          # only pool pages read
            rows = rows.reshape(len(want), p["out"], -1)
            if name == "down_proj":
                p["pool"] = mx.array(rows[:, :, :p["r"]])
                p["pool_s"] = p["s"][want_mx][:, :, :p["r"] // 4]
            else:
                p["pool"] = mx.array(rows[:, :p["r"], :])
                p["pool_s"] = p["s"][want_mx][:, :p["r"], :]
            # dynamic biases: members beta (exact with P1), rest the cold
            # floor bias (uniform 1.5s or empirical centroid, see __init__)
            b_dyn = p["b_cold"] * 1
            b_dyn[want_mx] = p["b"][want_mx]
            p["b_dyn"] = b_dyn
            mx.eval(p["pool"], p["pool_s"], p["b_dyn"], self.lookup)
        return len(entered)


def _p1_group_means(mm: np.memmap, out: int, cache: Path) -> np.ndarray:
    """Media por grupo-de-64 de los valores 2-bit del plano P1 (float16
    [experts, out, groups]). Una pasada por el memmap; cacheado a disco."""
    if cache.exists():
        return np.load(cache)
    experts = mm.shape[0]
    qbar = np.empty((experts, out, mm.shape[1] // out // 4), dtype=np.float16)
    lanes = 2 * np.arange(16, dtype=np.uint32)
    for ei in range(experts):
        w = np.asarray(mm[ei]).reshape(out, -1, 4)          # 4 words = grupo 64
        vals = (w[..., None] >> lanes) & 3                  # [out, g, 4, 16]
        qbar[ei] = vals.reshape(out, -1, 64).mean(-1)
    np.save(cache, qbar)
    return qbar


STATE: dict = {}


def patch_fast(model, art_dir: Path, pool_k: int,
               centroid: str = "uniform", p1_frac: str | float = 1.0) -> None:
    from mlx_lm.models import switch_layers as sl

    manifest = json.load(open(art_dir / "p1_manifest.json"))
    by_layer: dict[int, dict] = {}
    for _, meta in manifest.items():
        by_layer.setdefault(meta["layer"], {})[meta["proj"]] = meta

    blocks = find_moe_blocks(model)
    block_cls = type(blocks[0][1])
    layers = {id(b): FastLayer(b, art_dir, by_layer[li], pool_k, centroid,
                               p1_frac)
              for li, b in blocks}
    STATE.update({"layers": layers, "pending": {id(b): [] for _, b in blocks}})

    def patched(self, x):
        st = STATE["layers"][id(self)]
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1]) if x.ndim > 2 else x
        inds, scores = route(self, x_flat)   # family-agnostic; routing untouched
        if x_flat.shape[0] > 1:
            mx.eval(inds)   # prefill: pin now — lazy inds under memory
                            # pressure have produced garbage indices
        STATE["pending"][id(self)].append(inds)
        glu = self.switch_mlp

        if x_flat.shape[0] > 1:
            # EXACT PREFILL: the prompt is one batched pass — serve it at full
            # 4-bit (P1 for the union of needed experts read once from the
            # memmap). The pool policy applies to decode only, where task
            # quality is unaffected; this removes the teacher-forced/prefill
            # toll and gives decode an exact KV to start from.
            inds_np = np.array(inds)
            uniq, inv = np.unique(inds_np, return_inverse=True)
            rme = mx.array(inv.reshape(inds_np.shape).astype(np.int32))
            xxp = mx.expand_dims(x_flat, (-2, -3))

            def pfx(name, xin):
                p = st.projs[name]
                rows = mx.array(np.asarray(p["mm"][list(uniq)])).reshape(
                    len(uniq), p["out"], -1)
                s_u = p["s"][mx.array(uniq.astype(np.int32))]
                b_u = p["b"][mx.array(uniq.astype(np.int32))]
                return (mx.gather_qmm(xin, p["p0"], p["s"] * 4, p["b"],
                                      rhs_indices=inds, transpose=True,
                                      group_size=GROUP, bits=2)
                        + mx.gather_qmm(xin, rows, s_u, mx.zeros_like(s_u),
                                        rhs_indices=rme, transpose=True,
                                        group_size=GROUP, bits=2))

            h = glu.activation(pfx("up_proj", xxp), pfx("gate_proj", xxp))
            y = pfx("down_proj", h).squeeze(-2)
            y = (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
            return add_shared(self, x_flat, y).reshape(shape)

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
            if p["r"] == 0:
                return y
            xin1 = xin[..., : p["r"] * 16] if name == "down_proj" else xin
            y1 = mx.gather_qmm(xin1, p["pool"], p["pool_s"],
                               mx.zeros_like(p["pool_s"]), rhs_indices=remap,
                               transpose=True, group_size=GROUP, bits=2)
            if name != "down_proj" and p["r"] < p["out"]:
                y1 = mx.pad(y1, [(0, 0)] * (y1.ndim - 1) + [(0, p["out"] - p["r"])])
            return y + y1 * mask

        h = glu.activation(pf("up_proj", xx), pf("gate_proj", xx))
        y = pf("down_proj", h)
        if do_sort:
            y = sl._scatter_unsort(y, inv_order, inds.shape)
        y = y.squeeze(-2)
        y = (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
        return add_shared(self, x_flat, y).reshape(shape)

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
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 16384
    pages = {}
    for line in out.splitlines():
        if ":" in line:
            kk, v = line.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                pages[kk.strip()] = int(v)
    n = (pages.get("Pages free", 0) + pages.get("Pages inactive", 0)
         + pages.get("Pages purgeable", 0) + pages.get("Pages speculative", 0))
    return n * page / 1e9


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
