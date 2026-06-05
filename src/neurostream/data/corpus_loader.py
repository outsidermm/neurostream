"""MOABB dataset access for the open motor-imagery pretraining corpus.

Wraps the five supported MOABB datasets behind a uniform iterator that yields
one ``RecordingHandle`` per (subject, session, run).
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import mne
from moabb.datasets import (
    Lee2019_MI,
    PhysionetMI,
    Schirrmeister2017,
)

log = logging.getLogger(__name__)


# name -> MOABB dataset class.
DATASET_REGISTRY: dict[str, type] = {
    "PhysionetMI": PhysionetMI,
    "Lee2019_MI": Lee2019_MI,
    "Schirrmeister2017": Schirrmeister2017,
}


@dataclass(frozen=True)
class RecordingHandle:
    """One MOABB recording, flattened out of the nested get_data() dict."""

    source: str
    subject: int
    session: str
    run: str
    raw: mne.io.BaseRaw


def get_subjects(name: str) -> list[int]:
    """Return the full subject list for a registered MOABB dataset."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}; known: {list(DATASET_REGISTRY)}")
    return list(DATASET_REGISTRY[name]().subject_list)


def iter_dataset(
    name: str,
    subjects: list[int] | None = None,
) -> Iterator[RecordingHandle]:
    """Iterate (subject, session, run, raw) over a MOABB dataset.

    If ``subjects`` is ``None``, all subjects in the dataset are used. Subjects
    not present in the dataset are skipped with a warning; per-subject fetch
    failures are logged and skipped.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}; known: {list(DATASET_REGISTRY)}")
    dataset = DATASET_REGISTRY[name]()
    available = set(dataset.subject_list)
    use = list(dataset.subject_list) if subjects is None else list(subjects)

    for subj in use:
        if subj not in available:
            log.warning(f"{name}: subject {subj} not in dataset, skipping")
            continue
        try:
            tree = dataset.get_data(subjects=[subj])
        except Exception:
            log.exception(f"{name}: failed to fetch subject {subj}")
            continue

        # MOABB returns {subject: {session: {run: Raw}}}; flatten.
        for session_name, runs in tree.get(subj, {}).items():
            for run_name, raw in runs.items():
                yield RecordingHandle(
                    source=name,
                    subject=int(subj),
                    session=str(session_name),
                    run=str(run_name),
                    raw=raw,
                )
