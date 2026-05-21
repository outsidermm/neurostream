# tests/conftest.py
from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np
import pytest

from neurostream.data.channels import BCI_IV_2A_22_CHANNELS


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Root of the repo, where pyproject.toml lives."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def split_path(project_root: Path) -> Path:
    return project_root / "src" / "neurostream" / "data" / "bci_iv_2a_v1.json"


@pytest.fixture
def make_raw() -> Callable[..., mne.io.RawArray]:
    """Factory for synthetic MNE Raw objects.

    Returns a Raw carrying the 22 target channels plus extras. Data is white
    noise scaled so each channel's std is ``amplitude_uv`` microvolts.
    """

    def _make(
        seed: int = 0,
        fs: float = 500.0,
        n_seconds: float = 70.0,
        channels: tuple[str, ...] | None = None,
        amplitude_uv: float = 10.0,
        extra_channels: tuple[str, ...] = ("T7", "T8", "Fp1", "Fp2"),
    ) -> mne.io.RawArray:
        names = list(channels or BCI_IV_2A_22_CHANNELS) + list(extra_channels)
        rng = np.random.default_rng(seed)
        n_samples = int(fs * n_seconds)
        data = rng.standard_normal((len(names), n_samples)) * amplitude_uv * 1e-6
        info = mne.create_info(ch_names=names, sfreq=fs, ch_types="eeg")
        return mne.io.RawArray(data, info, verbose="ERROR")

    return _make
