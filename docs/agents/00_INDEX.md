# NeuroStream — Delivery Plan

This folder is the **forward-looking plan** for NeuroStream: what each phase
builds, the architectural decisions behind it, the ordered steps to execute,
and the constraints to execute it under. It is written for whoever (human or
AI agent) picks the project up and needs to know *what to do next* and *why the
design is the way it is* — not a status report.

Read in this order:

1. **[01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)** — what NeuroStream is,
   who it's for, the 5-phase roadmap, and what "done" means for each phase.
2. **[02_PHASE_2_PLAN.md](02_PHASE_2_PLAN.md)** — the active phase. Goal,
   definition of done, and the ordered, dependency-aware task list to finish
   it (loader fix → evaluation → fine-tune → ablation → release).
3. **[03_ARCHITECTURE_AND_DECISIONS.md](03_ARCHITECTURE_AND_DECISIONS.md)** —
   model configs, the hyperparameters the design commits to, and the reasoning
   behind the key choices.
4. **[04_DATA_PIPELINE.md](04_DATA_PIPELINE.md)** — the data architecture: the
   loaders that exist, the corpus harmonisation pipeline, and which loader is
   meant for which task.
5. **[05_ENVIRONMENT_AND_SETUP.md](05_ENVIRONMENT_AND_SETUP.md)** — hardware,
   tooling, and the OS/Hydra/MLflow constraints any execution step runs under.
6. **[06_PHASE_3_PLUS_PLAN.md](06_PHASE_3_PLUS_PLAN.md)** — intent for Phases
   3–5. Deliberately high-level: no day-by-day plan has been designed yet.

## Where current status lives

This folder describes *intent and design*. The point-in-time record of what is
actually done, in progress, or blocked lives separately in
**`docs/phase-2-notes/`**:

- [`phase-2-status.md`](../phase-2-notes/phase-2-status.md) — current status
  snapshot: what's built, what's blocked, what's left.
- [`probe-data-mismatch.md`](../phase-2-notes/probe-data-mismatch.md) — the
  linear-probe data-mismatch root cause + ablation, with error bars.

Keeping plan and status separate is deliberate: the plan should stay stable as
work progresses, and only the status files churn. If a step here is completed
or a decision here is made, record that in `phase-2-status.md`; only edit the
plan when the *intended design* changes.

## Source of truth

When this plan and the code disagree, the **code is the source of truth**.
These docs encode design intent and the rationale for it; if you find a
discrepancy, trust the code and fix the doc.
