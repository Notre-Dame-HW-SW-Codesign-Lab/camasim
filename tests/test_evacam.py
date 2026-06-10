"""Tests for the EvaCAM match backend (EVACAMConfig).

Skipped automatically when the compiled evacam_py extension is unavailable.
"""

import numpy as np
import pytest


def test_word_width_positive(evacam_config):
    assert evacam_config.word_width > 0


def test_exact_match_is_hit(evacam_config):
    row = np.zeros(evacam_config.word_width)
    matched, energy, latency = evacam_config.match(row, row)
    assert matched is True
    # A real search dissipates energy over a non-zero latency.
    assert energy > 0
    assert latency > 0


def test_single_bit_mismatch_is_miss(evacam_config):
    row = np.zeros(evacam_config.word_width)
    query = row.copy()
    query[0] = 1
    matched, _, _ = evacam_config.match(row, query)
    assert matched is False


def test_wrong_length_raises(evacam_config):
    row = np.zeros(evacam_config.word_width)
    short = np.zeros(evacam_config.word_width - 1)
    with pytest.raises(ValueError):
        evacam_config.match(row, short)


def test_write_is_passthrough(evacam_config):
    data = np.ones((4, evacam_config.word_width))
    out, energy, latency = evacam_config.write(data)
    assert np.array_equal(out, data)
    assert energy == 0.0
    assert latency == 0.0
