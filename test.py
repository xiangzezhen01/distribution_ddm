import random
from river import drift

rng = random.Random(12345)
kswin = drift.KSWIN(alpha=0.0001, seed=42)

data_stream = rng.choices([0, 1], k=100) + rng.choices(range(4, 8), k=100)

for i, val in enumerate(data_stream):
    kswin.update(val)
    if kswin.drift_detected:
        print(f"Change detected at index {i}, input value: {val}")