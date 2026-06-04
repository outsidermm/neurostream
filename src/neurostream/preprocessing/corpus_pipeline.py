"""Corpus harmonisation + ingest pipeline.

Ingests each configured MOABB dataset, runs every recording through the
preprocessing chain — channel selection -> resample -> bandpass -> common
average reference -> rejection — and writes the survivors to disk as
memory-mapped shards (~2GB each) plus metadata for each shard.

**Sharding during ingest** (vs. sharding after) halves peak storage by avoiding
an intermediate per-recording file stage. Critical for storage-constrained
environments.
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
from neurostream.data.corpus_loader import get_subjects, iter_dataset
from neurostream.utils.checkpoint import CheckpointManager
from neurostream.utils.git import git_sha
from neurostream.utils.io import atomic_save_npy, atomic_write_text

from .channel_selection import select_channels
from .filters import BandpassParams, bandpass_filter
from .missing_channels import ensure_channels
from .referencing import common_average_reference
from .resampling import resample_to_fs
from .source_scale import v_to_uv_scale

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

    raw = ensure_channels(raw, cfg.target_channels, source)
    data = select_channels(raw, cfg.target_channels, source)
    if data is None:
        return None, RejectionReason.MISSING_CHANNELS

    data = data * v_to_uv_scale(source)  # MNE volts -> microvolts
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
    """Harmonise every configured dataset and shard directly to disk.

    Shards recordings into ~2GB memory-mapped ``.npy`` files during ingestion,
    avoiding an intermediate per-recording stage. Halves peak storage vs. the
    ingest-then-shard approach.

    Layout written under ``out_dir``:
        shard_000.npy           # (n_channels, total_samples) concatenated
        shard_000_meta.json     # recording boundaries + provenance
        shard_001.npy
        shard_001_meta.json
        ...
        manifest.json           # top-level: all shards + rejected recordings
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    harmonise_cfg = harmonise_config_from_omegaconf(cfg.harmonise)
    shard_size_gb = cfg.get("shard_size_gb", 2.0)

    # Load any shards already on disk from a previous (interrupted) run.
    # This prevents overwriting existing data and lets the manifest be complete.
    existing_meta_paths = sorted(out_dir.glob("shard_*_meta.json"))
    all_shards: list[dict] = [
        json.loads(p.read_text()) for p in existing_meta_paths
    ]
    shard_idx = len(all_shards)

    # Shard buffering state
    shard_buffer: list[tuple[np.ndarray, dict]] = []  # (data, metadata) pairs
    shard_buffer_bytes = 0

    def flush_shard():
        """Write current shard buffer to disk and reset."""
        nonlocal shard_buffer, shard_buffer_bytes, shard_idx
        if not shard_buffer:
            return

        # Concatenate all recordings in buffer
        arrays = [data for data, _ in shard_buffer]
        concatenated = np.concatenate(arrays, axis=1)

        # Write shard
        shard_name = f"shard_{shard_idx:03d}.npy"
        atomic_save_npy(out_dir / shard_name, concatenated)

        # Build metadata: byte ranges for each recording
        recordings_in_shard = []
        offset = 0
        for data, meta in shard_buffer:
            n_samples = data.shape[1]
            recordings_in_shard.append(
                {
                    **meta,
                    "byte_offset": offset,
                    "n_samples": n_samples,
                }
            )
            offset += n_samples

        shard_meta = {
            "shard_name": shard_name,
            "shard_idx": shard_idx,
            "n_channels": int(concatenated.shape[0]),
            "total_samples": int(concatenated.shape[1]),
            "recordings": recordings_in_shard,
        }
        atomic_write_text(
            out_dir / f"shard_{shard_idx:03d}_meta.json",
            json.dumps(shard_meta, indent=2, default=str),
        )
        all_shards.append(shard_meta)

        log.info(
            f"Flushed shard_{shard_idx:03d}: "
            f"{len(recordings_in_shard)} recordings, "
            f"{concatenated.shape[1]} samples, "
            f"{concatenated.nbytes / 1e9:.2f} GB"
        )

        shard_buffer = []
        shard_buffer_bytes = 0
        shard_idx += 1

    checkpoint = CheckpointManager(out_dir / "checkpoint.json")
    # Seed rejected list from checkpoint so previous rejects survive a resume.
    rejected: list[dict] = checkpoint.rejected
    if checkpoint.completed_count:
        log.info(
            f"Resuming: {checkpoint.completed_count} subject(s) already done, "
            f"{len(all_shards)} shard(s) on disk"
        )

    # Ingest all datasets, one subject at a time so each can be checkpointed
    # after its shard is safely on disk.
    for ds_cfg in cfg.datasets:
        name = ds_cfg.name
        cfg_subjects = (
            list(ds_cfg.subjects) if ds_cfg.get("subjects") is not None else None
        )
        all_subjects = (
            cfg_subjects if cfg_subjects is not None else get_subjects(name)
        )
        log.info(f"=== {name}: {len(all_subjects)} subject(s) ===")

        for subj in all_subjects:
            if checkpoint.is_done(name, subj):
                log.debug(f"  {name}/{subj}: skip (checkpointed)")
                continue

            log.info(f"  {name}/{subj}: processing")
            subj_rejected: list[dict] = []
            yielded_any = False
            try:
                for handle in iter_dataset(name, [subj]):
                    yielded_any = True
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
                        subj_rejected.append({**base, "reason": "EXCEPTION"})
                        continue

                    if data is None:
                        assert reason is not None
                        subj_rejected.append({**base, "reason": reason.value})
                        continue

                    metadata = {
                        **base,
                        "n_channels": int(data.shape[0]),
                        "fs": harmonise_cfg.target_fs,
                        "units": "uV",
                    }
                    shard_buffer.append((data, metadata))
                    shard_buffer_bytes += data.nbytes

                    # Mid-subject flush only if a single subject exceeds shard limit
                    if shard_buffer_bytes > shard_size_gb * 1e9:
                        flush_shard()

            except Exception:
                log.exception(f"{name}/{subj}: fetch failed, skipping")
                continue

            # Flush after every subject so data is on disk before checkpointing.
            # mark_done also persists subj_rejected into the checkpoint so they
            # survive a future resume.
            flush_shard()
            rejected.extend(subj_rejected)
            if yielded_any or subj_rejected:
                checkpoint.mark_done(name, subj, subj_rejected)
                log.debug(f"  {name}/{subj}: checkpointed")
            else:
                log.warning(
                    f"  {name}/{subj}: iter_dataset yielded nothing and no rejections "
                    "— fetch may have failed silently, will retry on next run"
                )

    # Write top-level manifest
    manifest = {
        "version": "v2-sharded",
        "harmonise_config": asdict(harmonise_cfg),
        "shard_size_gb": shard_size_gb,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "shards": all_shards,
        "rejected": rejected,
        "total_recordings": sum(len(s["recordings"]) for s in all_shards),
        "total_shards": len(all_shards),
    }
    atomic_write_text(
        out_dir / "manifest.json", json.dumps(manifest, indent=2, default=str)
    )
    log.info(
        f"Done. kept={manifest['total_recordings']} "
        f"rejected={len(rejected)} "
        f"shards={manifest['total_shards']} "
        f"sources_seen={sorted(set(r['source'] for s in all_shards for r in s['recordings']))}"
    )
    return manifest
