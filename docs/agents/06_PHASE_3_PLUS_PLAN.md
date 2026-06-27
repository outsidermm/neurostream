# Phase 3+ Plan

**A statement of intent, not a spec.** Unlike Phases 1–2, no day-by-day plan
has been designed for Phases 3–5 yet. Everything below is the level of detail
actually decided so far — don't assume more specificity exists, and don't
fabricate day-by-day plans for phases that haven't been designed. The first real
task for any of these phases is writing its plan **with the user** (mirroring the
structure of the Phase 1/2 plans), not generating one unilaterally.

Hard sequencing constraint: **don't start Phase 3 work while Phase 2's loader
fix is open** (`02_PHASE_2_PLAN.md` step 1). Optimizing inference for a model
trained against out-of-distribution inputs wastes the effort twice.

## Phase 3: C++ SIMD-optimized inference engine

**Goal:** sub-10ms inference latency, framed for HFT/low-latency audiences as
much as ML ones.

**Decided so far:**
- AVX2 SIMD intrinsics, C++, CMake/Ninja build tooling.
- The fine-tuned MAE encoder + classification head (from Phase 2) gets
  ONNX-exported. The engine's correctness gate is byte-for-byte comparison
  against the PyTorch/ONNX output on Phase 1's test batches.
- **Phase 1's preprocessing was deliberately written as separately-tested pure
  functions** (not a class-based pipeline) precisely so the C++ port can be
  verified step-by-step against the Python reference, not just end-to-end. Don't
  refactor Phase 1 preprocessing into a different style without preserving this.
- The dev container already ships Phase 3's toolchain (`build-essential`,
  `cmake`, `clang-18`, `ninja-build`), added in Phase 1 to avoid invalidating
  Docker layers later.

**Critical dependency:** needs Phase 2's fine-tuned model correct and stable
first.

**Not yet decided:** day-by-day plan, specific SIMD kernels, exact ONNX export
procedure, parity-test tolerance thresholds, build structure beyond "CMake/Ninja."

## Phase 4: Kubernetes deployment + MLflow model registry

**Decided so far:**
- Docker image extends the Phase 1 dev container base (another reason that
  container was built with later phases in mind).
- The self-hosted MLflow server becomes the model registry; the Phase 2
  `v0.2.0` checkpoint becomes its first registered entry.
- Helm charts serve the model. CI extends with benchmark gates beyond Phase 1's
  basic three-check CI.
- Planned tooling: k3s/Helm, Terraform.

**Not yet decided:** everything beyond the above. No day-by-day plan.

## Phase 5: Observability, drift detection, SLOs

**Decided so far:**
- SLOs defined relative to baseline latency measurements (from Phase 3).
- Drift detection compares production prediction distributions against the
  Phase 1 evaluation distributions, and separately against the Phase 2
  pretraining corpus distribution.
- Planned tooling: Prometheus/Grafana, OpenTelemetry.

**Not yet decided:** everything beyond the above. No day-by-day plan.
