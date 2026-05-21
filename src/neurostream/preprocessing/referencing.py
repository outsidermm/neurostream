"""EEG re-referencing operations."""

import numpy as np


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the per-timestep mean across channels (CAR).

    Re-referencing to the common average makes recordings from different
    reference montages comparable.

    Args:
        data: (n_channels, n_samples).

    Returns:
        Re-referenced array, same shape.
    """
    return data - data.mean(axis=0, keepdims=True)
