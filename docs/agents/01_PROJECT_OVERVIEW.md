# Project Overview

## What this is

NeuroStream is a multi-phase ML engineering portfolio project. It demonstrates
a full research-to-production pipeline for EEG-based motor imagery
classification, built to be legible to three audiences: HFT/low-latency
systems roles, AI lab/ML research roles, and infrastructure/MLE roles.

**Dataset:** BCI Competition IV Dataset 2a. 9 subjects (A01–A09), 4-class
motor imagery (left hand / right hand / both feet / tongue), within-subject
protocol (train on session T, evaluate on session E), 22 EEG channels,
nominal 250 Hz sampling rate, ~288 trials per session.

**Portfolio framing (important — do not drift from this):** the project's
value proposition is engineering discipline and the research-to-production
gap, NOT research novelty or clinical relevance. Don't oversell pretraining
results as a research contribution; the honest framing is "I built the full
pipeline and understand every failure mode in it."

## The five-phase roadmap

| Phase | Content | Target outcome |
|---|---|---|
| 1 | Reproducible EEGNet supervised baseline | ~67–68% mean accuracy. **Done — tagged `v0.1.0`.** |
| 2 | Self-supervised MAE pretraining + linear probe + fine-tune | ≥3pp over Phase 1 (≥71% mean accuracy). **Active.** |
| 3 | C++ SIMD-optimized inference engine | Sub-10ms inference latency. |
| 4 | Kubernetes deployment + MLflow model registry | Reproducible serving + registered model. |
| 5 | Observability, drift detection, SLOs | Production monitoring against defined SLOs. |

Phases 1–2 are planned in detail in this folder (`02_PHASE_2_PLAN.md` for the
active phase). Phases 3–5 exist as high-level intent only — see
`06_PHASE_3_PLUS_PLAN.md`. Do not assume a detailed spec exists for them until
one has been written.

## Hardware constraint (summary — full detail in 05_ENVIRONMENT_AND_SETUP.md)

Windows native, PowerShell. GPU: **RTX 5060, 8GB VRAM** — a hard constraint
that shapes the plan: it has already forced deviations from published training
recipes (notably pretraining batch size; see `03_ARCHITECTURE_AND_DECISIONS.md`).
Do not assume more VRAM is available than this when proposing batch sizes,
model sizes, or anything else compute-bound.

## What "done" means for each phase

A colleague with Docker (Phase 1) or Docker + a CUDA GPU (Phase 2) should be
able to clone the repo, follow a short sequence of scripts, and reproduce the
reported numbers within a small tolerance, with no manual intervention.
**Reproducibility is the actual deliverable; the accuracy numbers are a
byproduct.** A slightly worse but fully reproducible result is preferred over a
better result that needed undocumented manual steps — keep this bar in mind
when deciding whether something is good enough to ship.
