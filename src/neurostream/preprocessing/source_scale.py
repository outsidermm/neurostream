"""Per-source V→µV scale corrections for harmonise().

Most MOABB datasets store EEG in volts (standard MNE convention), so multiplying
by 1e6 converts to microvolts.
"""

_DEFAULT_SCALE = 1e6


def v_to_uv_scale(source: str) -> float:
    """Return the V→µV multiplier for *source*."""
    return _DEFAULT_SCALE
