"""Tests for the corpus harmonisation + ingest pipeline.

Op-level behaviour (resampling, referencing, rejection, channel selection) is
covered by the dedicated test_*.py files. These tests cover the pipeline that
orchestrates them: harmonise() and ingest_corpus().
"""

import json

import mne
import numpy as np
import pytest
from omegaconf import OmegaConf

from neurostream.data.channels import BCI_IV_2A_22_CHANNELS
from neurostream.data.corpus_loader import DATASET_REGISTRY, RecordingHandle
from neurostream.preprocessing import corpus_pipeline
from neurostream.preprocessing.corpus_pipeline import (
    HarmoniseConfig,
    RejectionReason,
    check_rejection,
    harmonise,
    harmonise_config_from_omegaconf,
)

# A harmonise config block as it appears under `harmonise:` in Hydra.
_HARMONISE_CFG = {
    "target_fs": 128,
    "bandpass": [0.5, 45.0],
    "filter_order": 4,
    "reject_nan_frac": 0.1,
    "reject_amp_uv": 500.0,
    "reject_amp_frac": 0.05,
    "min_duration_s": 60.0,
}


# ---------------------------------------------------------------------------
# harmonise() — orchestration of the preprocessing chain
# ---------------------------------------------------------------------------


def test_harmonise_returns_correct_shape_and_dtype(make_raw):
    raw = make_raw(fs=500, n_seconds=70)
    out, reason = harmonise(raw, HarmoniseConfig())
    assert reason is None
    assert out is not None
    assert out.shape == (22, 70 * 128)
    assert out.dtype == np.float32


def test_harmonise_output_is_common_average_referenced(make_raw):
    raw = make_raw(fs=500, n_seconds=70)
    out, _ = harmonise(raw, HarmoniseConfig())
    assert out is not None
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-4)


def test_harmonise_is_deterministic(make_raw):
    raw = make_raw(fs=500, n_seconds=70, seed=42)
    out1, _ = harmonise(raw, HarmoniseConfig())
    out2, _ = harmonise(raw, HarmoniseConfig())
    np.testing.assert_array_equal(out1, out2)


def test_harmonise_does_not_modify_input_raw(make_raw):
    raw = make_raw(fs=500, n_seconds=70)
    names_before = list(raw.ch_names)
    sfreq_before = raw.info["sfreq"]
    harmonise(raw, HarmoniseConfig())
    assert list(raw.ch_names) == names_before
    assert raw.info["sfreq"] == sfreq_before


def test_harmonise_rejects_missing_channels(make_raw):
    raw = make_raw(fs=500, n_seconds=70)
    raw.drop_channels(["C3"])
    out, reason = harmonise(raw, HarmoniseConfig())
    assert out is None
    assert reason == RejectionReason.MISSING_CHANNELS


def test_harmonise_propagates_rejection_reason(make_raw):
    # A too-short recording must surface as a rejection from harmonise().
    raw = make_raw(fs=500, n_seconds=30)
    out, reason = harmonise(raw, HarmoniseConfig())
    assert out is None
    assert reason == RejectionReason.TOO_SHORT


# ---------------------------------------------------------------------------
# check_rejection — the pipeline's keep/drop quality gate
# ---------------------------------------------------------------------------


def _clean_uv(n_seconds: int = 70) -> np.ndarray:
    """Synthetic clean recording in µV, (22, n_seconds * 128)."""
    return np.random.default_rng(0).standard_normal((22, n_seconds * 128)) * 10.0


def test_check_rejection_passes_clean_recording():
    assert check_rejection(_clean_uv(), HarmoniseConfig()) is None


def test_check_rejection_flags_short_recording():
    out = check_rejection(_clean_uv(n_seconds=30), HarmoniseConfig())
    assert out == RejectionReason.TOO_SHORT


def test_check_rejection_flags_nan_heavy():
    data = _clean_uv()
    data[:, : int(0.15 * data.shape[1])] = np.nan  # 15% of all samples
    assert check_rejection(data, HarmoniseConfig()) == RejectionReason.NAN_HEAVY


def test_check_rejection_flags_amplitude_artifact():
    data = _clean_uv()
    data[5, :] += 2000.0  # channel 5 far beyond the 500 µV p2p threshold
    out = check_rejection(data, HarmoniseConfig())
    assert out == RejectionReason.AMPLITUDE_HEAVY


def test_check_rejection_checks_duration_first():
    # A recording both too short and all-NaN must report TOO_SHORT.
    data = _clean_uv(n_seconds=30)
    data[:] = np.nan
    assert check_rejection(data, HarmoniseConfig()) == RejectionReason.TOO_SHORT


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_harmonise_config_from_omegaconf_round_trips():
    cfg = harmonise_config_from_omegaconf(OmegaConf.create(_HARMONISE_CFG))
    assert cfg.target_fs == 128
    assert cfg.bandpass == (0.5, 45.0)
    assert cfg.target_channels == BCI_IV_2A_22_CHANNELS


def test_dataset_registry_contains_expected_sources():
    assert set(DATASET_REGISTRY) == {
        "PhysionetMI",
        "Lee2019_MI",
        "Schirrmeister2017",
    }


# ---------------------------------------------------------------------------
# ingest_corpus driver — exercised against synthetic data via monkeypatching
# ---------------------------------------------------------------------------


def test_ingest_corpus_writes_manifest_and_npy(tmp_path, make_raw, monkeypatch):
    def fake_iter(name, subjects):  # noqa: ARG001
        yield RecordingHandle(
            source=name,
            subject=1,
            session="0",
            run="0",
            raw=make_raw(fs=500, n_seconds=70),
        )

    monkeypatch.setattr(corpus_pipeline, "iter_dataset", fake_iter)

    cfg = OmegaConf.create(
        {
            "harmonise": _HARMONISE_CFG,
            "datasets": [{"name": "PhysionetMI", "subjects": [1]}],
        }
    )
    manifest = corpus_pipeline.ingest_corpus(cfg, tmp_path)

    assert manifest["version"] == "v2-sharded"
    assert manifest["total_recordings"] == 1
    assert manifest["total_shards"] == 1
    assert len(manifest["rejected"]) == 0

    shard = manifest["shards"][0]
    assert shard["n_channels"] == 22
    assert shard["total_samples"] == 70 * 128
    assert len(shard["recordings"]) == 1

    rec = shard["recordings"][0]
    assert rec["n_channels"] == 22
    assert rec["fs"] == 128
    assert rec["units"] == "uV"
    assert rec["n_samples"] == 70 * 128
    assert rec["byte_offset"] == 0

    arr = np.load(tmp_path / shard["shard_name"])
    assert arr.shape == (22, 70 * 128)
    assert arr.dtype == np.float32

    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["shards"] == manifest["shards"]


def test_ingest_corpus_records_rejections(tmp_path, make_raw, monkeypatch):
    def fake_iter(name, subjects):  # noqa: ARG001
        yield RecordingHandle(
            source=name,
            subject=1,
            session="0",
            run="0",
            raw=make_raw(fs=500, n_seconds=30),  # too short
        )

    monkeypatch.setattr(corpus_pipeline, "iter_dataset", fake_iter)

    cfg = OmegaConf.create(
        {
            "harmonise": _HARMONISE_CFG,
            "datasets": [{"name": "PhysionetMI", "subjects": [1]}],
        }
    )
    manifest = corpus_pipeline.ingest_corpus(cfg, tmp_path)

    assert manifest["total_recordings"] == 0
    assert manifest["total_shards"] == 0
    assert len(manifest["rejected"]) == 1
    assert manifest["rejected"][0]["reason"] == RejectionReason.TOO_SHORT.value


# ---------------------------------------------------------------------------
# Lee2019_MI FCz interpolation fix
# ---------------------------------------------------------------------------


def test_harmonise_lee2019_mi_passes_despite_missing_fcz():
    """Lee2019_MI lacks FCz; harmonise() must interpolate it and not reject."""
    channels_without_fcz = [ch for ch in BCI_IV_2A_22_CHANNELS if ch != "FCz"]
    rng = np.random.default_rng(0)
    n_samples = 70 * 500
    data = rng.standard_normal((len(channels_without_fcz), n_samples)) * 10e-6
    info = mne.create_info(ch_names=list(channels_without_fcz), sfreq=500.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    out, reason = harmonise(raw, HarmoniseConfig(), source="Lee2019_MI")
    assert reason != RejectionReason.MISSING_CHANNELS, (
        "harmonise() rejected Lee2019_MI for MISSING_CHANNELS; FCz interpolation not applied"
    )


# ---------------------------------------------------------------------------
# Silent-failure fix — subject not checkpointed when iter_dataset yields nothing
# ---------------------------------------------------------------------------


def test_ingest_corpus_does_not_checkpoint_subject_when_iter_yields_nothing(
    tmp_path, monkeypatch
):
    """When iter_dataset silently yields nothing (e.g. a caught fetch error),
    the subject must NOT be written to the checkpoint so the next run retries."""

    def fake_iter(name, subjects):  # noqa: ARG001
        return iter([])  # silent empty — simulates a caught OSError

    monkeypatch.setattr(corpus_pipeline, "iter_dataset", fake_iter)

    cfg = OmegaConf.create(
        {
            "harmonise": _HARMONISE_CFG,
            "datasets": [{"name": "PhysionetMI", "subjects": [1]}],
        }
    )
    corpus_pipeline.ingest_corpus(cfg, tmp_path)

    checkpoint_path = tmp_path / "checkpoint.json"
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text())
        assert ["PhysionetMI", 1] not in data.get("completed", []), (
            "Subject checkpointed despite iter_dataset yielding nothing"
        )


def test_ingest_corpus_checkpoints_subject_when_all_recordings_rejected(
    tmp_path, make_raw, monkeypatch
):
    """A subject whose recordings all fail rejection checks (subj_rejected non-empty)
    should still be checkpointed so the pipeline doesn't retry it needlessly."""

    def fake_iter(name, subjects):  # noqa: ARG001
        yield RecordingHandle(
            source=name,
            subject=1,
            session="0",
            run="0",
            raw=make_raw(fs=500, n_seconds=30),  # too short — will be rejected
        )

    monkeypatch.setattr(corpus_pipeline, "iter_dataset", fake_iter)

    cfg = OmegaConf.create(
        {
            "harmonise": _HARMONISE_CFG,
            "datasets": [{"name": "PhysionetMI", "subjects": [1]}],
        }
    )
    corpus_pipeline.ingest_corpus(cfg, tmp_path)

    checkpoint_path = tmp_path / "checkpoint.json"
    assert checkpoint_path.exists()
    data = json.loads(checkpoint_path.read_text())
    assert ["PhysionetMI", 1] in data["completed"]


# ---------------------------------------------------------------------------
# Slow test — exercises a real MOABB download
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_physionetmi_subject1_end_to_end(tmp_path):
    """Fetch one subject from PhysionetMI via MOABB, ingest, verify."""
    cfg = OmegaConf.create(
        {
            "harmonise": _HARMONISE_CFG,
            "datasets": [{"name": "PhysionetMI", "subjects": [1]}],
        }
    )
    manifest = corpus_pipeline.ingest_corpus(cfg, tmp_path)

    assert manifest["total_recordings"] > 0, (
        f"no recordings survived. rejected={manifest['rejected']}"
    )
    for shard in manifest["shards"]:
        arr = np.load(tmp_path / shard["shard_name"])
        assert arr.shape[0] == 22
        assert arr.dtype == np.float32
        for rec in shard["recordings"]:
            assert rec["fs"] == 128
