# filters.py
from dataclasses import dataclass
from typing import Tuple
from scipy.signal import butter, sosfiltfilt
import numpy as np


@dataclass(frozen=True)
class BandpassParams:
    low_hz: float
    high_hz: float
    fs_hz: float
    order: int = 4


def design_bandpass_sos(p: BandpassParams) -> Tuple | None:
    """Design a Butterworth bandpass filter, returned as second-order sections."""

    # Frequency is normalized to Nyquist frequency for digital filter design as it is the highest frequency that can be accurately represented at a given sampling rate.
    # Normalizing by Nyquist frequency ensures the filter design is correct for the digital domain.
    nyquist_freq = 0.5 * p.fs_hz
    if not (0 < p.low_hz < p.high_hz < nyquist_freq):
        raise ValueError(f"Invalid band [{p.low_hz}, {p.high_hz}] for fs={p.fs_hz}")

    high_norm_freq = p.high_hz / nyquist_freq
    low_norm_freq = p.low_hz / nyquist_freq

    # Second order sections are used for numerical stabilty for higher order filters
    return butter(p.order, [low_norm_freq, high_norm_freq], btype="band", output="sos")


def bandpass_filter(x: np.ndarray, p: BandpassParams) -> np.ndarray:
    """
    Zero-phase bandpass filter applied along the last axis.

    Args:
        x: array of shape (..., n_samples). Filtering is along axis=-1.
        p: filter parameters.

    Returns:
        Filtered array, same shape and dtype as input (cast to float64 internally
        for numerical stability, then back to input dtype).
    """
    sos = design_bandpass_sos(p)
    original_dtype = x.dtype

    # using forward pass and backward pass to eliminate phase delay.
    y = sosfiltfilt(sos, x.astype(np.float64, copy=False), axis=-1)
    return y.astype(original_dtype, copy=False)
