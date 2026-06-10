"""Shared fixtures for the camasim test suite.

EvaCAM-backed fixtures skip (rather than fail) when the compiled ``evacam_py``
extension or the default EvaCAM config is unavailable, so the pure-logic tests
still run in a minimal environment.
"""

import numpy as np
import pytest


def _evacam_config_path():
    from camasim.evacam import _DEFAULT_CONFIG

    return _DEFAULT_CONFIG


@pytest.fixture(scope="session")
def evacam_config():
    """Construct an EVACAMConfig, or skip if EvaCAM isn't available."""
    pytest.importorskip("evacam_py")
    if not _evacam_config_path().exists():
        pytest.skip("default EvaCAM config not found (evacam/ source tree absent)")
    from camasim.evacam import EVACAMConfig

    return EVACAMConfig()


@pytest.fixture(scope="session")
def word_width(evacam_config):
    return evacam_config.word_width


@pytest.fixture
def stored_data(word_width):
    """8 unique binary rows of the EvaCAM word width."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=(8, word_width)).astype(float)


@pytest.fixture
def written_cam(word_width, stored_data):
    """A CAMASim with stored_data written; returns (cam, data)."""
    from camasim import CAMASim, CAMConfig

    config = CAMConfig(subarray_rows=128, subarray_cols=word_width)
    cam = CAMASim(config)
    cam.write(stored_data)
    return cam, stored_data
