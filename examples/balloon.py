"""Memory-pressure balloon for the governor demo.

Run `serve.py --governor` in one terminal and this in another: it waits 25 s,
allocates 8 GB of touched pages, holds 45 s, releases. Watch the [gov] lines —
the measured behavior is stepwise K shrink under pressure and clean survival.
"""
import time

import numpy as np

print("[balloon] waiting 25 s…", flush=True)
time.sleep(25)
print("[balloon] INFLATING 8 GB", flush=True)
a = np.ones((8, 1024, 1024, 1024), dtype=np.uint8)
print("[balloon] holding 45 s", flush=True)
time.sleep(45)
del a
print("[balloon] deflated; holding 60 s", flush=True)
time.sleep(60)
print("[balloon] done", flush=True)
