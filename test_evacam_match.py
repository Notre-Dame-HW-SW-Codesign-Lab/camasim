"""Demo: CAMASim using EvaCAM's circuit-accurate match."""

import numpy as np

from camasim import CAMASim, CAMConfig

WORD = 64  # must equal the EvaCAM config's word_width

config = CAMConfig(
    subarray_rows=128,
    subarray_cols=WORD,
)

cam = CAMASim(config)

# 8 stored binary rows of width 64.
rng = np.random.default_rng(0)
data = rng.integers(0, 2, size=(8, WORD)).astype(float)

# Queries: rows 2 and 5 verbatim (should hit), plus one random row (likely miss).
queries = np.stack([data[2], data[5], rng.integers(0, 2, size=WORD).astype(float)])

write_res = cam.write(data)
query_res = cam.query(queries)

print(f"EvaCAM config : {cam.evacam.config_path}")
print(f"word_width    : {cam.evacam.word_width}")
print(f"write         : latency={write_res.latency} energy={write_res.energy}")
print(f"query latency : {query_res.latency:.4e} s   energy: {query_res.energy:.4e} J")
print(f"matched rows  : {query_res.indices}")

# Ground truth via numpy exact match.
expected = [sorted(np.where((data == q).all(axis=1))[0].tolist()) for q in queries]
print(f"expected rows : {expected}")
