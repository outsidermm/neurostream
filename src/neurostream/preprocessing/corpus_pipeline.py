"""Corpus harmonisation + ingest pipeline.

Ingests each configured MOABB dataset, runs every recording through the
preprocessing chain — channel selection -> resample -> bandpass -> common
average reference -> rejection — and writes the survivors to disk as
per-recording ``.npy`` files plus a ``manifest.json``.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import mne
import numpy as np
from omegaconf import DictConfig, OmegaConf

from neurostream.data.channels import BCI_IV_2A_22_CHANNELS
from neurostream.data.corpus_loader import iter_dataset
from neurostream.utils.git import git_sha
from neurostream.utils.io import atomic_save_npy, atomic_write_text
from neurostream.utils.naming import safe_filename

from .channel_selection import select_channels
from .filters import BandpassParams, bandpass_filter
from .referencing import common_average_reference
from .resampling import resample_to_fs

log = logging.getLogger(__name__)


class RejectionReason(str, Enum):
    """Why a recording was dropped during harmonisation."""

    MISSING_CHANNELS = "MISSING_CHANNELS"
    TOO_SHORT = "TOO_SHORT"
    NAN_HEAVY = "NAN_HEAVY"
    AMPLITUDE_HEAVY = "AMPLITUDE_HEAVY"


@dataclass(frozen=True)
class HarmoniseConfig:
    """Hyperparameters for the corpus harmonisation pipeline."""

    target_channels: tuple[str, ...] = BCI_IV_2A_22_CHANNELS
    target_fs: int = 128
    bandpass: tuple[float, float] = (0.5, 45.0)
    filter_order: int = 4
    reject_nan_frac: float = 0.10
    reject_amp_uv: float = 500.0
    reject_amp_frac: float = 0.05
    min_duration_s: float = 60.0


def harmonise_config_from_omegaconf(cfg: DictConfig) -> HarmoniseConfig:
    """Build a ``HarmoniseConfig`` from a Hydra/OmegaConf node."""
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    return HarmoniseConfig(
        target_channels=tuple(plain.get("target_channels", BCI_IV_2A_22_CHANNELS)),
        target_fs=int(plain["target_fs"]),
        bandpass=tuple(plain["bandpass"]),  # type: ignore[arg-type]
        filter_order=int(plain.get("filter_order", 4)),
        reject_nan_frac=float(plain["reject_nan_frac"]),
        reject_amp_uv=float(plain["reject_amp_uv"]),
        reject_amp_frac=float(plain["reject_amp_frac"]),
        min_duration_s=float(plain["min_duration_s"]),
    )


def check_rejection(data: np.ndarray, cfg: HarmoniseConfig) -> RejectionReason | None:
    """Return the first rejection reason that fires, or ``None`` if data passes.

    Rejection is a quality gate, not a signal transform — it decides whether a
    harmonised recording is kept. Checks run in order: duration, NaN fraction,
    amplitude. ``data`` is ``(n_channels, n_samples)`` in microvolts at
    ``cfg.target_fs``.
    """
    if data.shape[1] / cfg.target_fs < cfg.min_duration_s:
        return RejectionReason.TOO_SHORT
    if float(np.isnan(data).mean()) > cfg.reject_nan_frac:
        return RejectionReason.NAN_HEAVY
    p2p = data.max(axis=0) - data.min(axis=0)
    if float((p2p > cfg.reject_amp_uv).mean()) > cfg.reject_amp_frac:
        return RejectionReason.AMPLITUDE_HEAVY
    return None


def harmonise(
    raw: mne.io.BaseRaw,
    cfg: HarmoniseConfig,
    source: str = "",
) -> tuple[np.ndarray | None, RejectionReason | None]:
    """Run one recording through the preprocessing chain.

    Returns ``(array, None)`` on success — ``(n_channels, n_samples)`` float32
    in µV at ``cfg.target_fs`` — or ``(None, reason)`` if the recording is
    rejected.
    """
    source_fs = float(raw.info["sfreq"])

    data = select_channels(raw, cfg.target_channels, source)
    if data is None:
        return None, RejectionReason.MISSING_CHANNELS

    data = data * 1e6  # MNE volts -> microvolts
    data = resample_to_fs(data, source_fs, cfg.target_fs)
    data = bandpass_filter(
        data,
        BandpassParams(
            low_hz=cfg.bandpass[0],
            high_hz=cfg.bandpass[1],
            fs_hz=float(cfg.target_fs),
            order=cfg.filter_order,
        ),
    )
    data = common_average_reference(data)

    reason = check_rejection(data, cfg)
    if reason is not None:
        return None, reason
    return data.astype(np.float32, copy=False), None


def ingest_corpus(cfg: DictConfig, out_dir: Path) -> dict:
    """Harmonise every configured dataset to disk; return the manifest.

    Layout written under ``out_dir``:
        {source}/sub-{subject:03d}/ses-{session}_run-{run}.npy
        manifest.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    harmonise_cfg = harmonise_config_from_omegaconf(cfg.harmonise)

    recordings: list[dict] = []
    rejected: list[dict] = []

    for ds_cfg in cfg.datasets:
        name = ds_cfg.name
        subjects = list(ds_cfg.subjects) if ds_cfg.get("subjects") is not None else None
        log.info(f"=== {name}: subjects={subjects if subjects else 'all'} ===")
        try:
            for handle in iter_dataset(name, subjects):
                base = {
                    "source": handle.source,
                    "subject": handle.subject,
                    "session": handle.session,
                    "run": handle.run,
                }
                try:
                    data, reason = harmonise(handle.raw, harmonise_cfg, source=name)
                except Exception:
                    log.exception(f"{base}: harmonise() raised")
                    rejected.append({**base, "reason": "EXCEPTION"})
                    continue

                if data is None:
                    assert reason is not None
                    rejected.append({**base, "reason": reason.value})
                    continue

                rel_path = (
                    Path(handle.source)
                    / f"sub-{handle.subject:03d}"
                    / f"ses-{safe_filename(handle.session)}"
                    f"_run-{safe_filename(handle.run)}.npy"
                )
                atomic_save_npy(out_dir / rel_path, data)
                recordings.append(
                    {
                        **base,
                        "path": str(rel_path),
                        "n_channels": int(data.shape[0]),
                        "n_samples": int(data.shape[1]),
                        "fs": harmonise_cfg.target_fs,
                        "units": "uV",
                    }
                )
        except Exception:
            log.exception(f"{name}: source-level failure, continuing with next")
            continue

    manifest = {
        "version": "v1",
        "harmonise_config": asdict(harmonise_cfg),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "recordings": recordings,
        "rejected": rejected,
    }
    atomic_write_text(
        out_dir / "manifest.json", json.dumps(manifest, indent=2, default=str)
    )
    log.info(
        f"Done. kept={len(recordings)} rejected={len(rejected)} "
        f"sources_seen={sorted({r['source'] for r in recordings})}"
    )
    return manifest
