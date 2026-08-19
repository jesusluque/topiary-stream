"""Rondas intercaladas de tok/s nativo: original vs taper (mismo proceso NO —
proceso por medición para evitar contaminación de caché de Metal)."""
import sys
import time

from dense_loader import maybe_patch_per_layer  # PYTHONPATH del lab

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

model_path, name = sys.argv[1], sys.argv[2]
maybe_patch_per_layer(model_path)
model, tok = load(model_path)
mx.eval(model.parameters())
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Write a detailed essay about the history of computing."}],
    add_generation_prompt=True, tokenize=False)
n, t0 = 0, None
for r in stream_generate(model, tok, prompt, max_tokens=256,
                         sampler=make_sampler(temp=0.0)):
    if t0 is None:
        t0 = time.perf_counter()   # excluye el prefill: solo decode
    n += 1
dt = time.perf_counter() - t0
print(f"[{name}] {n - 1} tokens = {(n - 1) / dt:.1f} tok/s · pico {mx.get_peak_memory() / 1e9:.2f} GB")
