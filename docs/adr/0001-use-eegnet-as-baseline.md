# ADR 0001: Use EEGNet as the Phase 1 baseline model

## Status

Accepted — 2026-05-15. Implemented and validated by commit `f556545` (post-fix run `b199cad`, 0.692 mean 4-fold val accuracy).

## Context

Phase 1 of NeuroStream needs a published baseline for motor-imagery classification on BCI Competition IV Dataset 2a. The baseline serves three purposes beyond producing a number:

1. **Sanity check on the data pipeline.** If we can't reproduce a known published result, our preprocessing or label handling has a bug we need to fix before anything else.
2. **Reference point for Phase 2.** The masked-autoencoder model in Phase 2 is evaluated by linear probing — it has to be compared against *something* on the same dataset and same protocol.
3. **C++ port target for Phase 3.** Whatever model wins Phase 1 becomes the reference PyTorch implementation for byte-for-byte equivalence testing against the SIMD-vectorised C++ inference engine.

Constraints:

- Must have a published benchmark on BCI IV 2a (so we can verify our reproduction)
- Small parameter count (≤10k) so it trains on CPU/MPS in minutes, keeping CI loops fast
- Architectural simplicity — must export cleanly to ONNX for the Phase 3 port
- Active reference implementation we can cross-check against (the paper alone is not enough — see [phase-1-notes/04](../phase-1-notes/04-paper-faithful-reproduction.md))

## Decision

Use **EEGNet** (Lawhern et al., 2018) — specifically the EEGNet-8,2 configuration with `kernel_length=32` at 128 Hz — as the Phase 1 baseline.

Architecture: temporal conv → depthwise spatial conv → BN+ELU+AvgPool+Dropout → separable conv → BN+ELU+AvgPool+Dropout → flatten → linear. ~2,000 parameters. Implementation lives in [`src/neurostream/models/eegnet.py`](../../src/neurostream/models/eegnet.py).

## Alternatives considered

### ShallowConvNet (Schirrmeister et al., 2017)

- **Pros:** well-documented BCI IV 2a benchmark, equally simple architecture
- **Cons:** less faithful temporal-feature modelling, no widely-maintained reference Keras codebase to cross-check against
- **Rejected because:** we wanted both a paper AND a reference implementation. Two sources let us catch divergences (which is exactly how we found the missing `max_norm`)

### Deep4Net (Schirrmeister et al., 2017)

- **Pros:** higher reported accuracy (~73%)
- **Cons:** ~300k parameters, slower training, harder to interpret intermediate features for the Phase 2 MAE linear-probing setup
- **Rejected because:** overkill for a baseline whose purpose is *reproducibility*, not raw accuracy

### EEG-Conformer (Song et al., 2023)

- **Pros:** state-of-the-art accuracy (~78%)
- **Cons:** transformer architecture, ~1M parameters, no canonical PyTorch reference
- **Rejected because:** would conflate "is our baseline correct" with "is our transformer implementation correct" — too many unknowns at once. Possible Phase 2 comparison point

### FBCSP-style classical pipeline

- **Pros:** no deep-learning machinery, deterministic, fastest possible reference
- **Cons:** doesn't exercise the PyTorch training loop we need to debug for later phases
- **Rejected because:** Phase 1 needs to validate the full training stack, not just classification accuracy

## Consequences

**Positive:**
- Reproduced paper's mean 4-fold-CV accuracy within ~2 pp (achieved 0.692; paper reports ~71% for EEGNet-8,2)
- Small enough to train all 9 subjects × 4 folds × 300 epochs in ~30 min on M1 MPS
- Reading the [reference Keras code](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py) surfaced two `max_norm` constraints absent from the paper PDF — discovery documented in [phase-1-notes/04](../phase-1-notes/04-paper-faithful-reproduction.md)
- ONNX export is straightforward (only standard conv/BN/ELU/AvgPool/Linear ops) — de-risks Phase 3

**Negative:**
- Lower accuracy ceiling than transformer-based architectures (EEG-Conformer reports ~78% on the same dataset)
- Acceptable trade-off: Phase 2's MAE-pretrained model is where we aim to close the gap, using less labelled data

**Neutral:**
- Choice of EEGNet locks the Phase 2 MAE encoder into 2D-convolutional architecture for cross-comparability. Acceptable because the alternative (re-implementing the baseline in a different architecture) competes for attention with Phase 2 itself

## References

- Lawhern et al. (2018), *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces*, J. Neural Eng. 15(5). https://doi.org/10.1088/1741-2552/aace8c
- Reference implementation: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
- Our implementation: [`src/neurostream/models/eegnet.py`](../../src/neurostream/models/eegnet.py)
- Reproduction notes: [`docs/phase-1-notes/04-paper-faithful-reproduction.md`](../phase-1-notes/04-paper-faithful-reproduction.md)
