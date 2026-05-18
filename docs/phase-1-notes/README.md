# Phase 1 Lab Notes

*Engineering log of the obstacles, diagnostics, and fixes that produced the v0.1.0 EEGNet baseline on BCI IV 2a.*

## Headline

| Stage | Mean accuracy | Run ID | Protocol |
|---|---|---|---|
| Pre-fix baseline | **0.577** | `0282c73` | T→E single split, fs=250 |
| Post-fix baseline | **0.692** | `b199cad` | 4-fold CV on T, fs=128 |

Per-subject mean val accuracy (post-fix, run `b199cad`):

```
s01  0.767  ███████████████
s02  0.483  ██████████             ← weakest
s03  0.854  █████████████████
s04  0.545  ███████████
s05  0.573  ████████████
s06  0.549  ███████████
s07  0.764  ███████████████
s08  0.812  ████████████████
s09  0.882  ██████████████████     ← strongest
──────────────────────────
mean 0.692  ██████████████
```
*(each block ≈ 5 percentage points)*

## How to read these logs

Every log follows the same 5 sections so you can skim:

1. **TL;DR** — italicised single sentence under the title
2. **What we observed** — concrete failure mode
3. **What caused it** — root cause in plain English
4. **What we did** — fix + file path + commit SHA
5. **What this signals** — engineering principle demonstrated

## Index

| # | Title | TL;DR |
|---|---|---|
| 01 | [Window fix](01-window-fix.md) | TMAX 4.0 → 3.9 to keep all 288 trials |
| 02 | [EOG channels](02-eog-channels.md) | GDF marks every channel "eeg"; needed manual relabel |
| 03 | [Session E labels](03-session-e-labels.md) | Eval labels live in sibling `.mat`, not the GDF |
| 04 | [Paper-faithful reproduction](04-paper-faithful-reproduction.md) | `max_norm` + 4-fold CV — the headline win |
| 05 | [Resampling protocol](05-resampling-protocol.md) | 250→128 Hz + window [0.5, 2.5] s match Lawhern paper |
| 06 | [MPS numerical issues](06-mps-numerical-issues.md) | NaN val loss → no checkpoint → MLflow crash |
| 07 | [MLflow portability](07-mlflow-portability.md) | Container paths baked into experiment metadata |

## Audience

**Primary:** engineering recruiters scanning for evidence of debug skill, paper-reproduction discipline, and systems thinking. Read the index + log 04 for the headline; read 06 + 07 for examples of defensive-coding instincts.

**Secondary:** future-me debugging Phase 2.

## Cross-references

- Code path for everything: [`src/neurostream/`](../../src/neurostream/)
- MLflow runs: `mlruns/671296273026252269/` (post-fix), external `/Users/xjm/Desktop/mlruns/` (pre-fix)
- Branch: `xjm/eegnet`, commits referenced inline per log
