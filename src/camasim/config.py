from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union


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

    # Mapping scheme: how data/queries are laid out across the arrays.
    # Either a registered name ("default") or a (write_mapping, query_mapping)
    # pair of callables.
    mapping: Union[str, tuple[Callable, Callable]] = "default"

    # Merge strategies: either a registered name or a callable.
    # horizontal_merge: combines column-split results for one row ("and").
    # vertical_merge: combines matched row indices across rows ("union").
    horizontal_merge: Union[str, Callable] = "and"
    vertical_merge: Union[str, Callable] = "union"

    # Noise (None = no noise)
    noise: Optional[dict] = None  # {"type": "gaussian", "std": 0.1}
                                  # {"type": "bitflip", "rate": 0.05}
