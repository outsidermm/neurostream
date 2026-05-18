# 04 — Paper-faithful reproduction: `max_norm` + 4-fold CV

*The single biggest accuracy win in Phase 1. Reading the reference Keras code (not just the paper PDF) surfaced two constraints absent from the prose.*

## Headline

| Metric | Pre-fix | Post-fix |
|---|---|---|
| Mean accuracy | **0.577** | **0.692** |
| Run ID | `0282c73` | `b199cad` |
| Protocol | T→E single split | 4-fold CV on T |
| Model deltas | none | + `max_norm(1.0)` on depthwise conv, + `max_norm(0.25)` on dense |
| Sample rate | 250 Hz | 128 Hz (see [log 05](05-resampling-protocol.md)) |

> **Honest caveat:** the two headline numbers aren't strictly apples-to-apples. The old protocol scored on session E (cross-session generalisation); the new protocol scores on held-out folds of session T (within-session). Apples-to-apples val-on-T improved **0.646 → 0.692 (+4.6 pp)**; the larger headline gap reflects both better model fidelity AND an easier evaluation protocol. Both changes were intentional — the paper uses 4-fold CV, so we matched it.

## What we observed

- Initial baseline run (`0282c73`, pre-fix, fs=250) landed at 0.577 mean test acc — 10+ pp below Lawhern 2018's reported ~0.71
- Per-subject variance was extreme: s08 hit 0.91, s02 hit 0.38
- Overfit gate (`tests/test_overfit_gate.py`) still passed → model has capacity → it's a generalisation problem, not a wiring bug

## What caused it

Two divergences from the reference Keras implementation:

### 1. Missing `max_norm` weight constraints

Lawhern's [official Keras code](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py) applies two constraints that aren't called out in the paper PDF:

- `depthwise_constraint=max_norm(1.0)` on the spatial depthwise conv (block 1)
- `kernel_constraint=max_norm(0.25)` on the final dense layer

Keras rescales each filter's L2 norm after every optimiser step. PyTorch has no built-in equivalent. The constraints exist only in the code; reading the paper alone misses them. Most PyTorch ports of EEGNet that score 5–10 pp below paper numbers are missing exactly this.

### 2. Wrong evaluation protocol

The paper does **stratified 4-fold cross-validation on session T** and reports the mean across folds. Our first attempt did a single T→E split, which is a different (and harder) benchmark.

## What we did

- File: [`src/neurostream/models/eegnet.py`](../../src/neurostream/models/eegnet.py)
- File: [`src/neurostream/training/train.py`](../../src/neurostream/training/train.py)
- Commit: `f556545` — "feat: implement max norm constraint and 4 fold validation"

PyTorch port of Keras `max_norm`:

```python
def _renorm_max_norm(weight, max_value, dim):
    """Per-slice L2 max-norm; shrinks but never grows."""
    with torch.no_grad():
        norms = weight.norm(p=2, dim=dim, keepdim=True).clamp(min=1e-8)
        scale = (max_value / norms).clamp(max=1.0)
        weight.mul_(scale)

class EEGNet(nn.Module):
    def apply_max_norm(self):
        _renorm_max_norm(self.depthwise_conv1.weight, max_value=1.0, dim=(1, 2, 3))
        _renorm_max_norm(self.classifier.weight,      max_value=0.25, dim=1)
```

Training loop calls `model.apply_max_norm()` after every `optimizer.step()`. Detected at runtime via `getattr` so the loop stays model-agnostic.

Protocol change: replaced single `train_test_split` with `sklearn.model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)` over session T. Each fold gets its own per-fold pipeline fit (no leakage of normalisation statistics across folds).

## Per-subject val accuracy (apples-to-apples)

```
              pre    post    Δpp   delta
s01           0.71   0.77   +6.0   ▌▌▌
s02           0.38   0.48  +10.4   ▌▌▌▌▌
s03           0.83   0.85   +2.6   ▌
s04           0.55   0.55   -0.7   ▏
s05           0.43   0.57  +14.2   ▌▌▌▌▌▌▌    ← biggest gain
s06           0.45   0.55  +10.1   ▌▌▌▌▌
s07           0.69   0.76   +7.4   ▌▌▌▌
s08           0.91   0.81  -10.2   ▌▌▌▌▌ (-) ← only regression
s09           0.86   0.88   +2.0   ▌
─────────────────────────────────────────────────
mean          0.65   0.69   +4.6
```
*(each `▌` ≈ 2 percentage points; `▏` marks a near-zero delta)*

## What this signals

- **Paper PDF ≠ reference implementation.** Reading the authors' code is mandatory for faithful reproduction. The two missing `max_norm` constraints are the single most common reason published PyTorch EEGNet ports underperform Keras.
- **Protocol choice matters as much as model choice.** Both 0.577 and 0.692 are correct numbers; they measure different things. Reporting either without the other is misleading.
- **Ablation discipline matters.** Changing two things at once (model regularisation + evaluation protocol) makes the headline gap impossible to attribute. A cleaner experiment would have separated them — flagged as a Phase 2 follow-up.
- **Subject 8's regression is the most interesting data point.** A reproduction that improves the mean while *hurting* the strongest subject is suspicious — possibly the new stratified split disrupts a temporal structure subject 8 was exploiting. Worth investigating before claiming the change is unambiguously positive.
