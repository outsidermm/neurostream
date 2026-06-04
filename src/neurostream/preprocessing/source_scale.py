"""Per-source V→µV scale corrections for harmonise().

Most MOABB datasets store EEG in volts (standard MNE convention), so multiplying
by 1e6 converts to microvolts. Cho2017 is an exception: MOABB's gigadb.py applies
a *1e-6 factor assuming the .mat values are in µV, but the data is actually in
nanovolts (nV). This leaves MNE holding values 1000× too large (mV, not V).
Correcting with *1e3 instead of *1e6 restores physiological µV amplitudes.
"""

_PER_SOURCE_SCALE: dict[str, float] = {
    "Cho2017": 1e3,
}
_DEFAULT_SCALE = 1e6


def v_to_uv_scale(source: str) -> float:
    """Return the V→µV multiplier for *source*."""
    return _PER_SOURCE_SCALE.get(source, _DEFAULT_SCALE)
