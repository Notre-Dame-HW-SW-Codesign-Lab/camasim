"""Unit tests for the write/query mapping strategies (numpy only)."""

import numpy as np

from camasim._mapping import default_query_mapping, default_write_mapping


def test_write_mapping_shape_and_padding():
    data = np.ones((3, 4))
    padded = default_write_mapping(data, total_rows=8, col_splits=2, subarray_cols=4)

    assert padded.shape == (8, 8)  # total_rows x (col_splits * subarray_cols)
    # Original data sits in the top-left block.
    assert np.array_equal(padded[:3, :4], data)
    # Everything else is zero-padded.
    assert padded[3:].sum() == 0
    assert padded[:3, 4:].sum() == 0


def test_query_mapping_shape_and_split():
    queries = np.ones((2, 4))
    mapped = default_query_mapping(queries, col_splits=2, subarray_cols=4)

    assert mapped.shape == (2, 2, 4)  # N x col_splits x subarray_cols
    # First split holds the data, second split is padding.
    assert np.array_equal(mapped[:, 0, :], queries)
    assert mapped[:, 1, :].sum() == 0


def test_query_mapping_single_split_roundtrip():
    queries = np.array([[1.0, 0.0, 1.0]])
    mapped = default_query_mapping(queries, col_splits=1, subarray_cols=3)
    assert mapped.shape == (1, 1, 3)
    assert np.array_equal(mapped[0, 0], queries[0])
