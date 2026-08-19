# NeuroStream

**A research-to-production ML engineering portfolio project: EEG motor-imagery classification, built phase by phase from a reproducible baseline toward a low-latency inference engine.**

[![CI](https://img.shields.io/github/actions/workflow/status/outsidermm/neurostream/ci.yml?branch=main&label=CI)](https://github.com/outsidermm/neurostream/actions)
[![Python](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

NeuroStream classifies motor-imagery EEG signals (BCI Competition IV Dataset 2a). The point of the project isn't research novelty — it's demonstrating the full pipeline, end to end, and understanding every failure mode along the way: a paper-faithful supervised baseline, a self-supervised pretraining experiment (including where it fell short of target), and — planned — a low-latency C++ inference path and the MLOps scaffolding around it.

> **Status:** Phases 1–2 complete. See [Roadmap](#roadmap) for what's shipped vs. planned.

---

## What's actually built

**Phase 1 — EEGNet baseline (done, `v0.1.0`).** A PyTorch port of Lawhern et al. 2018's EEGNet, reproduced against the paper's reference Keras implementation (not just the paper PDF — matching its `max_norm` weight constraints was the single biggest accuracy fix). Stratified 4-fold CV on session T, evaluated on session E, within-subject protocol across all 9 subjects.

**Phase 2 — Self-supervised MAE pretraining (done, target missed, documented).** A masked-autoencoder transformer (He et al. 2022, adapted for EEG — temporal patches, 50% mask ratio) pretrained on an open EEG motor-imagery corpus, then evaluated by linear probe and end-to-end fine-tuning on BCI IV 2a:

- Linear probe: pretrained encoder beats a random-init control by **+14.18pp** (3-seed sweep), confirming the pretraining does transfer.
- Fine-tuning reached **61.50%** mean session-E accuracy — short of the ≥71% target, and below the EEGNet baseline. The miss is root-caused and written up (architecture mismatch, VRAM-constrained pretraining batch size, small per-subject labeled set) rather than left unexplained. See [`docs/phase-2-notes/finetune-results.md`](docs/phase-2-notes/finetune-results.md).

Full status snapshot: [`docs/phase-2-notes/phase-2-status.md`](docs/phase-2-notes/phase-2-status.md).

---

## Results

| Method | BCI IV 2a mean accuracy | Labeled data used |
|---|---|---|
| EEGNet baseline (Phase 1, this repo) | **69.2%** | 100% |
| Published EEGNet (Lawhern et al. 2018) | 71.1% | 100% |
| MAE linear probe vs. random-init control | +14.18pp gap | 100% (frozen encoder) |
| MAE + end-to-end fine-tune (Phase 2, this repo) | 61.50% | 100% |

Phase 1 detail and per-subject breakdown: [`docs/phase-1-notes/04-paper-faithful-reproduction.md`](docs/phase-1-notes/04-paper-faithful-reproduction.md). Phase 2 experiment log and honest post-mortem on the 71% miss: [`docs/phase-2-notes/finetune-results.md`](docs/phase-2-notes/finetune-results.md).

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Reproducible EEGNet supervised baseline | ✅ done — `v0.1.0` |
| 2 | Self-supervised MAE pretraining, linear probe, fine-tune | ✅ done (target missed, documented) |
| 3 | C++ SIMD-optimized inference engine, sub-10ms latency | ⏳ planned |
| 4 | Kubernetes deployment + MLflow model registry | ⏳ planned |
| 5 | Observability, drift detection, SLOs | ⏳ planned |

Phases 3–5 are intent-level only — no detailed day-by-day plan exists yet. See [`docs/agents/06_PHASE_3_PLUS_PLAN.md`](docs/agents/06_PHASE_3_PLUS_PLAN.md) for what's decided so far (AVX2 SIMD, CMake/Ninja, ONNX export + parity gate against Phase 1/2 outputs) versus what isn't.

---

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12. A CUDA GPU is recommended for pretraining/fine-tuning but not required for the baseline.

```bash
git clone https://github.com/outsidermm/neurostream.git
cd neurostream
uv sync --all-groups
```

Run the test suite:

```bash
uv run pytest
```

Phase 1 — train/evaluate the EEGNet baseline via Hydra:

```bash
uv run python -m neurostream.training.train
```

Phase 2 — pretrain the MAE, then probe or fine-tune it (see each script's docstring for Hydra overrides):

```bash
./scripts/pretrain.sh
uv run python -m scripts.linear_probe probe.pretrained_checkpoint=<path>
uv run python -m scripts.finetune finetune.pretrained_checkpoint=<path>
```

The dev container (`.devcontainer/`) pins the CUDA/Python/Clang toolchain, including the C++ build tools Phase 3 will use.

---

## Tech stack

| Layer | Tools |
|---|---|
| **ML** | PyTorch, Hydra, MNE-Python, MOABB, scikit-learn |
| **Experiment tracking** | MLflow |
| **Tooling** | `uv`, Ruff, mypy, pytest, pre-commit |
| **Dev environment** | Docker dev container (CUDA + Python 3.12 + Clang 18 — Clang/CMake pinned ahead of Phase 3) |
| **CI** | GitHub Actions — Ruff lint, mypy, pytest on every push/PR |
| **Planned (Phases 3–5)** | C++20 + AVX2 SIMD, ONNX Runtime, Kubernetes/Helm, Prometheus/Grafana |

---

## Repository layout

```
neurostream/
├── src/neurostream/
│   ├── data/              Corpus + BCI IV 2a loaders, windowing, harmonisation
│   ├── preprocessing/     Filtering, resampling, referencing, normalisation (pure functions —
│   │                      deliberately, so Phase 3 can port + verify them step-by-step in C++)
│   ├── models/            EEGNet, MAE encoder/decoder, patch + positional embeddings
│   ├── training/          Training loops: baseline, MAE pretraining, linear probe, fine-tune
│   └── eval/               Evaluation reporting
├── scripts/               CLI entry points (pretrain, fine-tune, linear probe, corpus ingestion)
├── configs/               Hydra configs (model, data, train, probe)
├── tests/                 Mirrors src/ layout — TDD throughout
├── notebooks/             Data sanity checks
├── docs/
│   ├── agents/            Forward-looking plan (phase specs, architecture decisions, setup)
│   ├── phase-1-notes/     Phase 1 debugging log (paper-faithful reproduction, etc.)
│   ├── phase-2-notes/     Phase 2 status snapshot, fine-tune results, ablations
│   └── adr/               Architecture decision records
└── .devcontainer/         Pinned CUDA + Python + Clang dev environment
```

---

## Documentation

Start at [`docs/agents/00_INDEX.md`](docs/agents/00_INDEX.md) for the full plan structure. Key pointers:

- [`docs/agents/01_PROJECT_OVERVIEW.md`](docs/agents/01_PROJECT_OVERVIEW.md) — what this project is and who it's for
- [`docs/phase-2-notes/phase-2-status.md`](docs/phase-2-notes/phase-2-status.md) — current point-in-time status
- [`docs/adr/`](docs/adr/) — architecture decisions (why MAE over contrastive methods, why EEGNet as baseline, etc.)
- [`docs/phase-1-notes/`](docs/phase-1-notes/) and [`docs/phase-2-notes/`](docs/phase-2-notes/) — debugging journals

When plan docs and code disagree, the code is the source of truth — see the note at the top of [`docs/agents/00_INDEX.md`](docs/agents/00_INDEX.md).

---

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

BCI Competition IV Dataset 2a (Graz University of Technology) and the EEGNet authors (Lawhern et al., 2018).
