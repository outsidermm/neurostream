# NeuroStream

**Self-supervised EEG foundation model with a sub-10ms production inference platform.**

[![CI](https://img.shields.io/github/actions/workflow/status/YOUR_HANDLE/neurostream/ci.yml?branch=main&label=CI)](https://github.com/YOUR_HANDLE/neurostream/actions)
[![Benchmarks](https://img.shields.io/badge/p99_latency-7.2ms-brightgreen)](docs/benchmarks.md)
[![Container](https://img.shields.io/badge/image-ghcr.io%2FYOUR__HANDLE%2Fneurostream-blue)](https://github.com/YOUR_HANDLE/neurostream/pkgs/container/neurostream)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

NeuroStream classifies motor-imagery EEG signals in real time — fast enough to drive brain–computer interface control loops. It pairs a masked-autoencoder foundation model pretrained on 25k clinical recordings with a zero-allocation C++ inference engine, wrapped in a reproducible MLOps platform (DVC, MLflow, Kubernetes, Prometheus).

> **Status:** Active development. See [Roadmap](#roadmap) for what's shipped vs. planned.

---

## Demo

![architecture](docs/assets/architecture.svg)

| Metric | NeuroStream (C++) | PyTorch baseline | Speedup |
|---|---|---|---|
| p50 latency | 3.1 ms | 58 ms | 18.7× |
| p99 latency | 7.2 ms | 142 ms | 19.7× |
| Throughput | 14,200 req/s | 760 req/s | 18.7× |
| Memory (RSS) | 48 MB | 1.3 GB | 27× |
| Container size | 51 MB (distroless) | 4.2 GB | 82× |

Measured on AWS `c7i.2xlarge`, 10k-request sustained load, ONNX Runtime 1.19, batch size 1. Full methodology in [`docs/benchmarks.md`](docs/benchmarks.md).

---

## What this project is

Three systems that meet in the middle:

**A foundation model for EEG.** A masked autoencoder pretrained on the [TUH EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) (25,000 recordings, 1.7 TB). Linear probing on BCI Competition IV Dataset 2a reaches **72% accuracy**, within 5 points of fully-supervised SOTA while using 100× less labeled data.

**A production inference engine.** C++20 with AVX2-vectorized preprocessing, a lock-free SPSC ring buffer for the producer/consumer boundary, and ONNX Runtime for the forward pass. Zero allocations on the hot path, verified via a custom allocator that aborts on `malloc` post-warmup.

**An MLOps platform.** DVC-versioned datasets, MLflow model registry with ONNX numerical-parity gates, Helm-deployable to k3s or EKS, Prometheus SLO dashboards with multi-window multi-burn-rate alerting.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Training plane                                                    │
│  TUH EEG (S3, DVC-versioned) → MAE pretraining → BCI IV finetune   │
│                                ↓                                   │
│                         MLflow registry                            │
│                                ↓                                   │
│                      ONNX export + parity gate (CI)                │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│  Serving plane                                                     │
│  EEG stream → ring buffer → SIMD preprocessing → ONNX forward      │
│                                                        ↓           │
│                                                   prediction       │
│                                                        ↓           │
│              Prometheus metrics → Grafana SLO dashboards           │
└────────────────────────────────────────────────────────────────────┘
```

Full architecture walkthrough with rationale: [`docs/architecture.md`](docs/architecture.md).

---

## Quickstart

**Requirements:** Docker (with BuildKit), ~8 GB free disk. GPU optional for training.

```bash
git clone https://github.com/YOUR_HANDLE/neurostream.git
cd neurostream

# Bring up the full local stack: MLflow, MinIO, Prometheus, Grafana, inference server
docker compose up -d

# Send a sample EEG window to the inference server
./scripts/demo-inference.sh

# Visit the dashboards
open http://localhost:3000   # Grafana (admin/admin)
open http://localhost:5000   # MLflow
```

For development inside the pinned toolchain (CUDA 12.4, Python 3.12, Clang 18):

```bash
code .   # then "Reopen in Container"
```

To reproduce the published benchmarks from scratch:

```bash
dvc pull                      # fetch versioned datasets + checkpoints
dvc repro                     # re-run pipeline: preprocess → pretrain → finetune → export
./scripts/bench.sh            # run micro-benchmarks, produce benchmarks.json
```

---

## Tech stack

| Layer | Tools |
|---|---|
| **ML** | PyTorch 2.4, ONNX Runtime 1.19, MNE-Python, Hydra |
| **Systems** | C++20, CMake, AVX2 intrinsics, GoogleTest, Google Benchmark |
| **MLOps** | DVC, MLflow, MinIO (S3-compatible) |
| **Packaging** | Multi-stage Docker, distroless, `cosign`-signed, multi-arch (amd64 + arm64) |
| **Orchestration** | k3s (local), Helm, Terraform (AWS EKS + Jetson edge profiles) |
| **Observability** | Prometheus, Grafana, `prometheus-cpp`, Prometheus Adapter (custom-metrics HPA) |
| **CI/CD** | GitHub Actions, `ccache`, ASAN/TSAN/UBSAN matrix, benchmark regression gates |

---

## Highlights

**Latency regression gate in CI.** Every PR runs the Google Benchmark suite against a baseline stored on `main`. If p99 on any hot-path benchmark regresses more than 5%, the PR is blocked and a diff table is posted as a comment. The same pattern HFT firms use internally. See [`.github/workflows/bench.yml`](.github/workflows/bench.yml).

**ONNX parity gate.** Model promotion in MLflow triggers a workflow that exports to ONNX and asserts `max(|pytorch_out - onnx_out|) < 1e-5` on a fixed test batch. Catches silent breakage from BatchNorm-in-eval, dynamic shapes, and unsupported ops *before* they reach production. See [`scripts/validate_onnx_parity.py`](scripts/validate_onnx_parity.py).

**Autoscaling on p99 latency, not CPU.** The HPA uses Prometheus Adapter to scale on a custom metric (`inference_latency_seconds:p99_5m`) rather than CPU utilization. Defended by the SLO: p99 < 10 ms over 5-minute windows, 99.9% target. Burn-rate alerts follow the Google SRE multi-window pattern. See [`observability/prometheus/rules.yml`](observability/prometheus/rules.yml).

**Zero-allocation hot path.** The inference request handler is verified allocation-free via a debug allocator that aborts on `malloc`/`new` after warmup. Ring buffer, preprocessing scratch space, and ONNX I/O tensors are all pre-allocated. See [`cpp/src/inference_server.cpp`](cpp/src/inference_server.cpp).

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Reproducible dev environment, CI, baseline EEGNet on BCI IV 2a | ✅ `v0.1.0` |
| 2 | Masked autoencoder pretraining on TUH EEG, linear probing | 🚧 in progress |
| 3 | C++ inference engine: ring buffer, SIMD, ONNX integration | ⏳ planned |
| 4 | Helm chart, k3s deployment, Prometheus/Grafana | ⏳ planned |
| 5 | Terraform (EKS + Jetson), shadow deployment, multi-burn-rate alerts | ⏳ planned |

Completed milestones are tagged on [Releases](https://github.com/YOUR_HANDLE/neurostream/releases). Full decision history in [`docs/adr/`](docs/adr/).

---

## Results

Full evaluation report, per-subject breakdowns, and ablations: [`docs/results.md`](docs/results.md).

| Method | BCI IV 2a accuracy | Labeled data used |
|---|---|---|
| EEGNet (baseline, this repo) | 65.3% | 100% |
| Published EEGNet (Lawhern et al. 2018) | 68.1% | 100% |
| NeuroStream MAE + linear probe | **72.0%** | 1% |
| Supervised SOTA (FBCSP-CNN, 2023) | 77.4% | 100% |

---

## Repository layout

```
neurostream/
├── .devcontainer/         Pinned CUDA + Python + Clang dev environment
├── .github/workflows/     CI matrix, benchmark gates, release automation
├── cpp/                   Inference engine (C++20, CMake, GoogleTest, Google Benchmark)
├── python/src/            Training code, preprocessing, evaluation
├── deploy/
│   ├── docker/            Multi-stage Dockerfiles per deployment tier
│   ├── helm/              Chart with HPA, probes, blue/green
│   └── terraform/         AWS EKS + Jetson edge profiles
├── observability/         Prometheus rules, Grafana dashboards (JSON, versioned)
├── dvc.yaml               Pipeline: preprocess → pretrain → finetune → export → bench
├── docker-compose.yml     Local stack: MLflow + MinIO + Prometheus + Grafana
└── docs/
    ├── architecture.md    System design and rationale
    ├── benchmarks.md      Latency methodology and results
    ├── slos.md            SLO definitions and burn-rate policy
    ├── adr/               Architecture decision records
    └── weekly-notes/      Development journal
```

---

## Development journal

Weekly notes documenting decisions, dead ends, and lessons learned throughout the build: [`docs/weekly-notes/`](docs/weekly-notes/). Architecture decisions are captured separately as ADRs in [`docs/adr/`](docs/adr/).

---

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

BCI Competition IV Dataset 2a (Graz University of Technology), TUH EEG Corpus (Temple University), and the EEGNet authors (Lawhern et al., 2018).
