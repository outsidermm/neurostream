"""Probe data path that matches the pretraining corpus harmonisation.

The MAE encoder was pretrained on the open corpus harmonised to 128 Hz with a
0.5-45 Hz band-pass and common-average reference (see
``preprocessing/corpus_pipeline.harmonise``). The Phase 1 BCI IV 2a loader does
NOT band-pass or re-reference, so its windows are out-of-distribution for the
encoder. Per-window z-scoring (the only normalisation the probe applied) is a
per-channel affine scale — it cannot undo a spatial CAR or a spectral filter.

This module loads each subject's *continuous* recording and applies the exact
corpus chain (optionally — toggled for the ablation), then hands the continuous
array to ``window_extract`` so windows can be sliced without zero-padding.

Channel order: the BCI IV 2a GDF exposes its 22 EEG channels in Brunner-montage
physical order, whose named anchors (Fz@0, C3@7, Cz@9, C4@11, Pz@19) coincide
exactly with ``BCI_IV_2A_22_CHANNELS``. So ``picks="eeg"`` already yields the
encoder's positional channel order; the other 17 GDF names are generic
(``EEG-0``..``EEG-16``) and cannot be matched by name.
"""

import functools
from typing import Literal

import mne
import numpy as np
import scipy.io

from neurostream.data.bci_iv_loader import (
    CUE_TEST_ID,
    CUE_TRAIN_IDS,
    DATA_RAW,
    EOG_CHANNELS,
    TARGET_SFREQ,
    _DATASET,
)
from neurostream.data.window_extract import centered_windows, padded_windows
from neurostream.preprocessing.filters import BandpassParams, bandpass_filter
from neurostream.preprocessing.referencing import common_average_reference
from neurostream.preprocessing.resampling import resample_to_fs
from neurostream.preprocessing.source_scale import v_to_uv_scale

# Corpus harmonisation hyperparameters (configs/pretrain_corpus.yaml).
CORPUS_BANDPASS = (0.5, 45.0)
CORPUS_FILTER_ORDER = 4


def _cue_samples_and_labels(
    raw: mne.io.BaseRaw,
    subject_id: int,
    session: Literal["T", "E"],
    source_fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cue sample indices (resampled to TARGET_SFREQ) and class labels (0..3)."""
    events, eid = mne.events_from_annotations(raw, verbose="ERROR")
    if session == "T":
        # Cue code encodes the class directly.
        inverse = {v: CUE_TRAIN_IDS[k] for k, v in eid.items() if k in CUE_TRAIN_IDS}
        mask = np.isin(events[:, 2], list(inverse))
        cue_src = events[mask, 0]
        labels = np.array([inverse[c] for c in events[mask, 2]], dtype=np.int64)
    else:
        code = eid[CUE_TEST_ID]
        cue_src = events[events[:, 2] == code, 0]
        label_path = DATA_RAW / _DATASET / f"A0{subject_id}E.mat"
        classlabel = scipy.io.loadmat(label_path)["classlabel"].squeeze()
        labels = classlabel.astype(np.int64) - 1  # 1..4 -> 0..3

    cue_128 = np.round(cue_src * TARGET_SFREQ / source_fs).astype(np.int64)
    if cue_128.shape != labels.shape:
        raise ValueError(
            f"A0{subject_id}{session}: {len(cue_128)} cues vs {len(labels)} labels"
        )
    return cue_128, labels


@functools.lru_cache(maxsize=64)
def load_continuous(
    subject_id: int,
    session: Literal["T", "E"],
    harmonise: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one continuous session as ``(data, cue_samples, labels)``.

    ``data`` is ``(22, n_times)`` float32 at TARGET_SFREQ. When ``harmonise`` is
    True the corpus band-pass + CAR are applied (matching pretraining); when
    False only resampling is done (reproduces the original probe distribution).
    Cached because the probe loads each session twice (pretrained + random).
    """
    path = DATA_RAW / _DATASET / f"A0{subject_id}{session}.gdf"
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    raw.set_channel_types({name: "eog" for name in EOG_CHANNELS})
    source_fs = float(raw.info["sfreq"])

    cue_128, labels = _cue_samples_and_labels(raw, subject_id, session, source_fs)
    if len(cue_128) != 288:
        raise AssertionError(f"expected 288 trials, got {len(cue_128)}")

    data = raw.get_data(picks="eeg")  # (22, n) volts, Brunner = canonical order
    if data.shape[0] != 22:
        raise AssertionError(f"expected 22 EEG channels, got {data.shape[0]}")

    data = data * v_to_uv_scale("")  # V -> uV (washed out by z-score; kept faithful)
    data = resample_to_fs(data, source_fs, TARGET_SFREQ)
    if harmonise:
        data = bandpass_filter(
            data,
            BandpassParams(
                low_hz=CORPUS_BANDPASS[0],
                high_hz=CORPUS_BANDPASS[1],
                fs_hz=float(TARGET_SFREQ),
                order=CORPUS_FILTER_ORDER,
            ),
        )
        data = common_average_reference(data)

    return data.astype(np.float32), cue_128, labels


def make_probe_adapter(
    harmonise: bool,
    window: Literal["pad2s", "pad4s", "continuous"],
    n_samples: int = 1000,
):
    """Build a ``(subject_id, session) -> (epochs, labels)`` probe adapter.

    Args:
        harmonise: apply the corpus band-pass + CAR (True) or not (False).
        window: ``"pad2s"`` reproduces the 2 s zero-padded window; ``"continuous"``
            slices ``n_samples`` real samples centred on each cue.
    """

    def adapter(
        subject_id: int, session: Literal["T", "E"]
    ) -> tuple[np.ndarray, np.ndarray]:
        data, cue, labels = load_continuous(subject_id, session, harmonise)
        if window == "continuous":
            epochs, n_padded = centered_windows(data, cue, n_samples=n_samples)
            if n_padded:
                # Surface any edge clamping instead of letting it pass silently.
                import logging

                logging.getLogger(__name__).warning(
                    "A0%d%s: %d/%d continuous windows hit a recording edge "
                    "and were zero-padded",
                    subject_id, session, n_padded, len(cue),
                )
        elif window in ("pad2s", "pad4s"):
            seconds = 2.0 if window == "pad2s" else 4.0
            epochs = padded_windows(
                data, cue, window_seconds=seconds, n_samples=n_samples
            )
        else:
            raise ValueError(f"unknown window mode: {window}")
        return epochs, labels

    return adapter
