"""
EvaCAM-backed cost/match model for CAMASim.

This wraps the compiled ``evacam_py`` pybind11 module (the ``evacam-py``
dependency, built from the EvaCAM C++ project) so CAMASim can use EvaCAM's
circuit-accurate match evaluation. For each row/query pair EvaCAM returns
whether the row is a hit plus the search latency and dynamic energy from its
circuit model.

The default match config is shipped with the ``evacam-py`` distribution (the
``evacam_assets`` package), so no EvaCAM source checkout is needed; a local
``evacam/`` source tree is used as a development fallback.
"""

import contextlib
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Local EvaCAM source tree (development fallback). This file lives at
# <repo>/src/camasim/evacam.py, so the repo root is parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVACAM_DIR = _REPO_ROOT / "evacam"
_SOURCE_CONFIG = _EVACAM_DIR / "config" / "2FeFET_TCAM" / "2FeFET_TCAM_match_system_config.yaml"


def _default_config_and_root():
    """Return ``(config_path, chdir_root)`` for the default match config.

    Prefers the config bundled in the installed ``evacam_assets`` package
    (so no EvaCAM source checkout is needed); falls back to the local
    ``evacam/`` source tree for development. EvaCAM resolves a config's
    ``cell_file`` relative to the working directory, so ``chdir_root`` is the
    directory those relative paths are anchored to.
    """
    try:
        import evacam_assets

        if evacam_assets.DEFAULT_MATCH_CONFIG.exists():
            return evacam_assets.DEFAULT_MATCH_CONFIG, evacam_assets.CONFIG_ROOT
    except ImportError:
        pass
    return _SOURCE_CONFIG, _EVACAM_DIR


# Resolved once at import; _DEFAULT_CONFIG is part of the public-ish surface
# (tests check whether it exists to decide whether to run EvaCAM-backed cases).
_DEFAULT_CONFIG, _DEFAULT_CONFIG_ROOT = _default_config_and_root()


def _load_evacam_module():
    """Import the compiled evacam_py module.

    Prefers an installed module (the CI-built package). Falls back to a local
    source-tree build in ``evacam/`` for development.
    """
    try:
        import evacam_py  # noqa: E402  (installed package)
    except ImportError:
        if str(_EVACAM_DIR) not in sys.path:
            sys.path.insert(0, str(_EVACAM_DIR))
        try:
            import evacam_py  # noqa: E402  (local source build)
        except ImportError as exc:  # pragma: no cover - build guidance
            raise ImportError(
                "Could not import the compiled 'evacam_py' module. Install the "
                f"package (built on CI) or build it locally from {_EVACAM_DIR}."
            ) from exc
    return evacam_py


@contextlib.contextmanager
def _chdir(path):
    """Temporarily change CWD (EvaCAM configs use evacam/-relative paths)."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class EVACAMConfig:
    """EvaCAM circuit-model backend for CAMASim.

    Parameters
    ----------
    config_path:
        Path to an EvaCAM system-config YAML. Defaults to the shipped
        2FeFET TCAM match config (64-bit word width). The matcher's
        ``word_width`` must equal CAMASim's ``subarray_cols``.
    """

    def __init__(self, config_path: Optional[str] = None):
        evacam_py = _load_evacam_module()
        if config_path is None:
            self.config_path = str(_DEFAULT_CONFIG)
            # Anchor the config's relative cell_file path to the bundled (or
            # source) config root.
            root = _DEFAULT_CONFIG_ROOT
        else:
            # Caller-supplied config: its cell_file is resolved from the
            # current working directory.
            self.config_path = str(config_path)
            root = None
        ctx = _chdir(root) if root is not None and root.is_dir() else contextlib.nullcontext()
        with ctx:
            self._matcher = evacam_py.EvaCAMMatch(self.config_path)
        self.word_width = self._matcher.word_width()

    @staticmethod
    def _to_binary(vec) -> list:
        """Coerce a numeric vector into a binary int list for EvaCAM."""
        arr = np.rint(np.asarray(vec)).astype(int)
        return arr.tolist()

    def match(self, row, query) -> tuple[bool, float, float]:
        """Evaluate one stored row against one query column-split.

        Returns ``(matched, energy_J, latency_s)`` from EvaCAM's circuit model.
        """
        if len(row) != self.word_width or len(query) != self.word_width:
            raise ValueError(
                f"EvaCAM word_width is {self.word_width} but got row/query of "
                f"length {len(row)}/{len(query)}. Set subarray_cols == "
                f"{self.word_width} (or load a matching EvaCAM config)."
            )
        result = self._matcher.evaluate_vector(self._to_binary(row), self._to_binary(query))
        return bool(result.hit), float(result.search_dynamic_energy), float(result.search_latency)

    def write(self, data) -> tuple:
        """Pass-through write.

        EvaCAM's match app models the search path only, so no write-side
        variation/cost is applied here. Returns ``(data, energy, latency)``.
        """
        return data, 0.0, 0.0
