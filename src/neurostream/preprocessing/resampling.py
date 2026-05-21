"""Resample EEG signals to a target sample rate.

Polyphase filtering (`scipy.signal.resample_poly`) preserves biosignal
fidelity better than FFT-based resampling.
"""

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def resample_to_fs(data: np.ndarray, source_fs: float, target_fs: int) -> np.ndarray:
    """Polyphase-resample ``data`` along the last axis to ``target_fs``.

    Args:
        data: (..., n_samples) signal.
        source_fs: current sample rate; must be an integer rate.
        target_fs: desired sample rate.

    Returns:
        Resampled array. Returned unchanged if the rates already match.
    """
    if abs(source_fs - target_fs) < 1e-3:
        return data
    if abs(source_fs - round(source_fs)) > 1e-6:
        raise ValueError(f"Non-integer source fs {source_fs} not supported")
    # All supported datasets have integer rates rationally related to the
    # target, so Fraction yields exact small up/down ratios.
    ratio = Fraction(target_fs, int(round(source_fs)))
    return resample_poly(data, up=ratio.numerator, down=ratio.denominator, axis=-1)
