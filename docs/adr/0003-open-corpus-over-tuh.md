# ADR 0003: Aggregated open motor-imagery corpus over TUH EEG

## Status
Accepted

## Context
Phase 2 pretraining needs a large unlabeled EEG corpus. Two approaches:
(a) Temple University Hospital (TUH) EEG Corpus (~1.7TB, dominantly clinical
seizure data, requires institutional registration), or (b) aggregate several
freely-available open motor-imagery datasets via MOABB.

## Decision
Aggregate five open motor-imagery datasets via MOABB:

| Dataset | Subjects | Channels | Native Hz |
|---|---|---|---|
| PhysionetMI | 109 | 64 | 160 |
| Cho2017 | 52 | 64 | 512 |
| Lee2019_MI | 54 | 62 | 1000 |
| Stieger2021 | 62 | 64 | 1000 |
| Schirrmeister2017 | 14 | 128 | 500 |

~290 subjects total. Harmonised to 22 channels (matching BCI IV 2a montage),
bandpass 0.5–45 Hz, common average reference (CAR), sharded into ~2GB
memory-mapped numpy files.

## Alternatives considered
- **TUH EEG Corpus**: much larger (1.7TB vs. ~80GB). Rejected for three
  reasons: (1) requires institutional registration, making the pipeline
  non-reproducible by an arbitrary reader with just a GitHub account;
  (2) dominantly clinical seizure data — weaker task-family match to
  motor imagery than our aggregate; (3) using a single pre-cleaned source
  avoids the multi-source harmonisation engineering that's arguably the most
  impressive and realistic part of the data pipeline.

## Consequences
**Positive:**
- Fully reproducible: anyone with GitHub access can rerun the entire pipeline
  end-to-end.
- Same task family (motor imagery) as the downstream evaluation — tighter
  pretrain-to-downstream alignment than TUH clinical data.
- Forces real multi-source data engineering: 5 datasets with different
  sampling rates (160–1000 Hz), channel counts (62–128), and reference
  schemes. The harmonisation pipeline (resampling, channel selection/
  reordering, filtering, re-referencing, rejection criteria) is more
  representative of real-world data work than ingesting one curated source.

**Negative:**
- Smaller total size (~80GB vs. 1.7TB). Accepted — the task-family match
  compensates for raw volume, and we're compute-constrained (8GB VRAM, batch
  64) anyway.
- Narrower clinical diversity — may underrepresent pathological EEG patterns.
  Accepted as irrelevant for this downstream task (motor imagery
  classification, not clinical diagnosis).

**Unexpected consequence (discovered during evaluation):**
- The harmonisation pipeline's preprocessing steps (bandpass + CAR) turned
  out to be load-bearing for downstream transfer. When the linear probe
  evaluation omitted these steps, the pretrained-vs-random gap collapsed
  from ~13pp to ~3-5pp. This is documented in ADR 0004 and the probe data
  mismatch investigation report.
