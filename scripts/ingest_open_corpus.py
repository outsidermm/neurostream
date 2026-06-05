"""Hydra CLI driver for the Phase 2 open-corpus ingest.

Usage:
    uv run python scripts/ingest_open_corpus.py
    uv run python scripts/ingest_open_corpus.py datasets=[{name:PhysionetMI,subjects:[1]}]

Wall time on a fresh machine is dominated by MOABB's first-run download
(~50-80 GB total across the four datasets, multiple hours). MOABB caches to
``paths.raw_cache`` (default: data/raw) so subsequent runs only re-harmonise.
CPU time for harmonisation alone is roughly 20-40 min depending on disk speed.
"""

import logging
import os
from pathlib import Path

import hydra
import mne
from omegaconf import DictConfig

from neurostream.preprocessing.corpus_pipeline import ingest_corpus

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(
    version_base=None,
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="pretrain_corpus",
)
def main(cfg: DictConfig) -> None:
    # Point MNE/MOABB at the project-local raw cache before any dataset access.
    # Both the env var and the persisted MNE config must agree — MOABB falls
    # back to the config file for fresh downloads and ignores the env var alone.
    raw_cache = (_PROJECT_ROOT / cfg.paths.raw_cache).resolve()
    os.environ["MNE_DATA"] = str(raw_cache)
    mne.set_config("MNE_DATA", str(raw_cache))

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    out_dir = (_PROJECT_ROOT / cfg.paths.output).resolve()
    log = logging.getLogger("ingest")
    log.info(f"raw cache: {raw_cache}")
    log.info(f"writing to {out_dir}")
    manifest = ingest_corpus(cfg, out_dir)
    log.info(
        f"kept={manifest['total_recordings']} "
        f"rejected={len(manifest['rejected'])} "
        f"manifest={out_dir / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
