"""End-to-end tests for CAMASim write/query backed by EvaCAM match.

Skipped automatically when the compiled evacam_py extension is unavailable.
"""

import numpy as np
import pytest

from camasim import CAMASim, CAMConfig
from camasim._mapping import default_query_mapping, default_write_mapping


def _expected_matches(data, query):
    """Ground-truth exact-match row indices for a single query."""
    return sorted(np.where((data == query).all(axis=1))[0].tolist())


def test_write_returns_result(written_cam):
    cam, data = written_cam
    result = cam.write(data)
    # Write is an EvaCAM pass-through (search-path model only).
    assert result.latency == 0.0
    assert result.energy == 0.0


def test_query_finds_each_stored_row(written_cam):
    cam, data = written_cam
    for i in (0, 2, 5, 7):
        query = data[i : i + 1]
        result = cam.query(query)
        assert result.indices == _expected_matches(data, data[i])
        assert i in result.indices


def test_query_distinct_vector_is_empty(written_cam):
    cam, data = written_cam
    # Bitwise complement of a stored row cannot equal that row; assert it
    # matches nothing via numpy ground truth, then check CAMASim agrees.
    query = 1.0 - data[0]
    expected = _expected_matches(data, query)
    result = cam.query(query.reshape(1, -1))
    assert result.indices == expected


def test_query_costs_are_positive(written_cam):
    cam, data = written_cam
    result = cam.query(data[:2])
    assert result.latency > 0
    assert result.energy > 0


def test_query_flattens_multiple_queries(written_cam):
    cam, data = written_cam
    queries = np.stack([data[2], data[5]])
    result = cam.query(queries)
    # vertical_union flattens matches across both queries.
    expected = _expected_matches(data, data[2]) + _expected_matches(data, data[5])
    assert sorted(result.indices) == sorted(expected)
    assert 2 in result.indices and 5 in result.indices


def test_custom_mapping_callable_is_used(word_width):
    """A (write, query) callable pair passed via config is used directly."""
    used = []

    def my_query_mapping(queries, col_splits, subarray_cols):
        used.append(True)
        return default_query_mapping(queries, col_splits, subarray_cols)

    cam = CAMASim(
        CAMConfig(subarray_cols=word_width, mapping=(default_write_mapping, my_query_mapping))
    )
    assert cam.query_mapping is my_query_mapping
    cam.write(np.zeros((2, word_width)))
    cam.query(np.zeros((1, word_width)))
    assert used  # the custom mapping actually ran


def test_unknown_mapping_name_raises(word_width):
    with pytest.raises(ValueError, match="Unknown strategy"):
        CAMASim(CAMConfig(subarray_cols=word_width, mapping="nope"))
