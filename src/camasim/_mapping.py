"""
Mapping strategies for writing and querying CAM arrays.
"""

import numpy as np


def default_write_mapping(data: np.ndarray, total_rows: int, col_splits: int, subarray_cols: int) -> np.ndarray:
    """Pad and map data into (total_rows, col_splits * subarray_cols)."""
    N, D = data.shape
    padded = np.zeros((total_rows, col_splits * subarray_cols))
    padded[:N, :D] = data
    return padded


def default_query_mapping(queries: np.ndarray, col_splits: int, subarray_cols: int) -> np.ndarray:
    """Pad and split queries into (N, col_splits, subarray_cols)."""
    N, D = queries.shape
    padded = np.zeros((N, col_splits * subarray_cols))
    padded[:, :D] = queries
    return padded.reshape(N, col_splits, subarray_cols)
