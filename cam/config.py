from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class CAMConfig:
    # Search
    distance: Literal["hamming", "l1", "l2"] = "hamming"
    match: Literal["exact", "best", "threshold"] = "exact"
    match_param: Optional[float] = None  # k for best, threshold value for threshold

    # Architecture hierarchy
    num_banks: int = 1
    num_mats: int = 1
    num_subarrays: int = 1
    subarray_rows: int = 128
    subarray_cols: int = 128

    # Merge strategies
    horizontal_merge: Literal["and"] = "and"
    vertical_merge: Literal["union"] = "union"

    # Noise (None = no noise)
    noise: Optional[dict] = None  # {"type": "gaussian", "std": 0.1}
                                  # {"type": "bitflip", "rate": 0.05}
