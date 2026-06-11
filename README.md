# CAMASim

A simulation framework for content-addressable memory (CAM) based accelerators.
It evaluates both functional behavior (which rows match a query) and hardware
cost (search latency and energy), using [EvaCAM](https://github.com/Notre-Dame-HW-SW-Codesign-Lab/evacam)
for circuit-accurate match evaluation.

## Install

```sh
uv sync
```

This installs the `cam` package and pulls the `evacam-py` extension (EvaCAM's
pybind11 bindings) from git. Building the extension needs a C++17 toolchain;
on macOS also `brew install yaml-cpp libomp`.

## Usage

```python
import numpy as np
from camasim import CAMASim, CAMConfig

# subarray_cols must equal the EvaCAM config's word_width (64 for the default).
config = CAMConfig(subarray_rows=128, subarray_cols=64)
cam = CAMASim(config)

data = np.random.randint(0, 2, (8, 64)).astype(float)
queries = np.stack([data[2], data[5]])

cam.write(data)
result = cam.query(queries)           # QueryResult(indices, latency, energy)
print(result.indices, result.latency, result.energy)
```

See `tests/` for runnable end-to-end examples.

## Layout

- `src/camasim/` — the simulation package: `CAMASim` (`write`/`query`),
  `CAMConfig`, pluggable mapping/merge strategies, and `EVACAMConfig` wrapping
  EvaCAM's match.
- `evacam/` — the EvaCAM C++ project (built and installed as the `evacam-py`
  dependency).
