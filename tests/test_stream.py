"""Pure-logic tests: no model downloads, no GPU-heavy work.

Covers the arithmetic the runtime rests on (plane packing, centroid floor,
anchored truncation), the pool state machine with synthetic memmaps, the
governor thresholds, and the model-card placeholder gate.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import GROUP, pack, pack2, unpack4, unpack_bits  # noqa: E402

# ------------------------------------------------------------------- planes


def _rand_qtensor(e=4, out=8, inp=128, bits=4, seed=0):
    mx.random.seed(seed)
    w = mx.random.normal((e, out, inp)).astype(mx.float32) * 0.05
    wq, s, b = mx.quantize(w, group_size=GROUP, bits=bits)
    mx.eval(wq, s, b)
    return w, wq, s, b


def test_unpack_pack_roundtrip():
    _, wq, _, _ = _rand_qtensor()
    codes = unpack4(wq)
    assert codes.max() <= 15 and codes.min() >= 0
    hi, lo = codes >> 2, codes & 3
    assert (hi * 4 + lo == codes).all()
    for packed, ref in ((pack2(hi), hi), (pack2(lo), lo)):
        words = np.array(packed, copy=True).view(np.uint32)
        cols = [(words >> (2 * j)) & 3 for j in range(16)]
        got = np.stack(cols, axis=-1).reshape(ref.shape)
        assert (got == ref).all()


def test_plane_reconstruction_exact_through_kernel():
    """P0+P1 through the real kernel == matmul against the original 4-bit."""
    w, wq, s, b = _rand_qtensor()
    codes = unpack4(wq)
    p0, p1 = pack2(codes >> 2), pack2(codes & 3)
    x = mx.random.normal((2, 1, 1, 128)).astype(mx.float32)
    idx = mx.array([[1], [3]])
    kw = dict(rhs_indices=idx, transpose=True, group_size=GROUP)
    y4 = mx.gather_qmm(x, wq, s, b, bits=4, **kw)
    y_planes = (mx.gather_qmm(x, p0, s * 4, b, bits=2, **kw)
                + mx.gather_qmm(x, p1, s, mx.zeros_like(s), bits=2, **kw))
    mx.eval(y4, y_planes)
    assert float(mx.abs(y4 - y_planes).max()) < 1e-4


def test_cold_centroid_bias():
    """The cold path (P0 with beta+1.5s) == truncated dequant with centroid."""
    _, wq, s, b = _rand_qtensor()
    codes = unpack4(wq)
    p0 = pack2(codes >> 2)
    wd_cold = mx.dequantize(p0[0], s[0] * 4, b[0] + 1.5 * s[0],
                            group_size=GROUP, bits=2)
    manual = (codes[0] >> 2).astype(np.float32).reshape(8, -1, GROUP)
    sn = np.array(s[0], dtype=np.float32)[..., None]
    bn = np.array(b[0], dtype=np.float32)[..., None]
    ref = (manual * (sn * 4) + bn + 1.5 * sn).reshape(8, -1)
    assert np.abs(np.array(wd_cold) - ref).max() < 1e-5


def test_anchored_q8_truncation_recovers_q4():
    """The anchored level-8 truncated >>4 must recover the Q4 anchor exactly."""
    mx.random.seed(3)
    w = mx.random.normal((8, 128)).astype(mx.float32)
    q4, s4, b4 = mx.quantize(w, group_size=GROUP, bits=4)
    mx.eval(q4, s4, b4)
    c4 = unpack_bits(q4, 4).astype(np.float32)
    wn = np.array(w, dtype=np.float32)
    s4n = np.repeat(np.array(s4, dtype=np.float32), GROUP, axis=-1)
    b4n = np.repeat(np.array(b4, dtype=np.float32), GROUP, axis=-1)
    r = wn - (s4n * c4 + b4n)
    q_lo = np.clip(np.round((r + s4n / 2) / (s4n / 16) - 0.5), 0, 15)
    codes8 = (c4 * 16 + q_lo).astype(np.uint16)
    assert (codes8 >> 4 == c4.astype(np.uint16)).all()
    w8 = codes8.astype(np.float32) * (s4n / 16) + (b4n - 15 * s4n / 32)
    w4 = s4n * c4 + b4n
    assert np.abs(wn - w8).mean() < np.abs(wn - w4).mean()


def test_unpack_bits_order():
    codes = np.arange(64, dtype=np.uint8).reshape(2, 32) % 16
    assert (unpack_bits(pack(codes, 4), 4) == codes).all()


# --------------------------------------------------------- pool state machine


def _synthetic_pool_layer(tmp_path, n_experts=16, out=8, inp=128,
                          pool_c=6, pool_k=3):
    import pager

    rng = np.random.default_rng(0)
    manifest = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        o, i = (out, inp) if proj != "down_proj" else (inp, out * 8)
        cols = i
        words = o * cols // 16
        rng_codes = rng.integers(0, 4, (n_experts, words), dtype=np.uint32)
        rng_codes.tofile(tmp_path / f"L0.{proj}.p0.bin")
        rng_codes.tofile(tmp_path / f"L0.{proj}.p1.bin")
        groups = cols // GROUP if cols % GROUP == 0 else 2
        sb = rng.standard_normal((n_experts, 2, o * groups)).astype(np.float16)
        sb.tofile(tmp_path / f"L0.{proj}.sb.bin")
        manifest[f"L0.{proj}"] = {"experts": n_experts, "out": o, "cols": cols,
                                  "words": words, "sb_cols": o * groups,
                                  "s_shape": [o, groups], "layer": 0, "proj": proj}
    prior = np.arange(n_experts, 0, -1).astype(np.float64)
    return pager.PoolLayer(0, tmp_path, manifest, prior, pool_c, pool_k)


def test_pool_init_and_lookup(tmp_path):
    st = _synthetic_pool_layer(tmp_path)
    assert len(st.members0) == 6 and len(st.members1) == 3
    assert st.members0 == [0, 1, 2, 3, 4, 5]        # prior-ordered
    assert set(st.members1) <= set(st.members0)
    l0 = np.array(st.lookup0)
    assert (l0[st.members0] >= 0).all()
    assert (l0[[10, 15]] == -1).all()


def test_pool_refresh_incremental_and_churn_cap(tmp_path):
    import pager

    st = _synthetic_pool_layer(tmp_path)
    counts = np.zeros(16)
    counts[[10, 11, 12, 13, 14, 15]] = 100          # the tail gets hot
    st.refresh(counts)
    entered = set(st.members0) & {10, 11, 12, 13, 14, 15}
    assert 0 < len(entered) <= pager.MAX_CHURN
    l0 = np.array(st.lookup0)
    for e in st.members0:
        assert l0[e] >= 0
    assert (np.array(st.lookup0) >= 0).sum() == len(st.members0)


def test_pool_p1_subset_invariant(tmp_path):
    st = _synthetic_pool_layer(tmp_path)
    counts = np.zeros(16)
    counts[[7, 8, 9]] = 50
    st.refresh(counts)
    assert set(st.members1) <= set(st.members0)


# --------------------------------------------------------------- governor


def test_governor_thresholds(monkeypatch, tmp_path):
    import fastpath

    class _FakeLayer:
        def __init__(self):
            self.pool_k = 32
            self.members = [1, 2]

    fake = _FakeLayer()
    fastpath.STATE["layers"] = {0: fake}
    monkeypatch.setattr(fastpath, "available_gb", lambda: 2.0)
    msg = fastpath.govern(low=4.0, high=7.0, step=8)
    assert "32->24" in msg and fake.pool_k == 24
    assert fake.members == []                        # forces rebuild
    monkeypatch.setattr(fastpath, "available_gb", lambda: 9.0)
    msg = fastpath.govern(low=4.0, high=7.0, step=8, k_max=48)
    assert "24->32" in msg
    monkeypatch.setattr(fastpath, "available_gb", lambda: 5.0)
    assert fastpath.govern(low=4.0, high=7.0) is None


# ------------------------------------------------------------- model cards


def test_model_card_placeholder_gate():
    import re as _re

    template = Path(__file__).parent.parent / "huggingface" / "MODEL_CARD_TEMPLATE_STREAM.md"
    if not template.exists():
        pytest.skip("template not written yet")
    text = template.read_text()
    assert _re.search(r"\{[A-Z][A-Z0-9_]*\}", text), \
        "template must contain placeholders for the upload gate to catch"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
