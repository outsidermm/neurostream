"""Tests for preprocessing.source_scale — per-source V→µV scale factors."""

from neurostream.preprocessing.source_scale import v_to_uv_scale


def test_default_scale_is_1e6():
    assert v_to_uv_scale("PhysionetMI") == 1e6
    assert v_to_uv_scale("Lee2019_MI") == 1e6
    assert v_to_uv_scale("Schirrmeister2017") == 1e6
    assert v_to_uv_scale("") == 1e6
    assert v_to_uv_scale("unknown_dataset") == 1e6
