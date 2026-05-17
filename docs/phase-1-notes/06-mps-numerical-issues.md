# 06 — Apple MPS numerical issues: NaN losses on subject 2

*Run crashed with `FileNotFoundError`. Root cause was IEEE 754 NaN semantics + MPS BatchNorm instability + a missing defensive write.*

## What we observed

- Full 9-subject training run crashed midway through subject 2 fold 0:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '/Users/xjm/Desktop/neurostream/outputs/checkpoints/subject_02_fold0_best.pt'
```

- Subject 1 finished cleanly. Subject 2 fold 0 completed all 300 epochs but produced no checkpoint.
- MLflow's `log_artifact` was the line that crashed — the *symptom*, not the cause.

## What caused it

Three things stacked:

### 1. MPS + BatchNorm + small batch produced NaN val loss

Apple Silicon MPS backend is newer than CUDA and has known numerical edge cases for tiny-batch BatchNorm. Subject 2's random model init (RNG state advanced by subject 1's prior training) landed in one such case → forward pass produced non-finite logits → val loss = NaN.

### 2. IEEE 754: `nan < anything == False`

The "save best checkpoint" guard was:

```python
best_val_loss = float("inf")
...
if val_m.loss < best_val_loss:           # nan < inf → False
    torch.save(model.state_dict(), ckpt_path)
```

NaN compares as False to every other value. With NaN val loss for every epoch → guard never triggered → checkpoint file never created.

### 3. MLflow `log_artifact` doesn't check for file existence

It just calls `mkdir + copy` and surfaces a `FileNotFoundError` on the source path — a confusing message that hides the upstream "no checkpoint produced" failure.

## What we did

- File: [`src/neurostream/training/train.py`](../../src/neurostream/training/train.py) (within `_train_fold`)

Two defensive changes:

```python
# Always have *some* checkpoint on disk — random init is a useless but valid fallback
torch.save(model.state_dict(), ckpt_path)

nan_epochs = 0
for epoch in range(1, cfg.training.epochs + 1):
    train_m = run_epoch(model, train_loader, criterion, optimizer, device)
    val_m   = run_epoch(model, val_loader,   criterion, None,      device)

    if not np.isfinite(val_m.loss):
        nan_epochs += 1                  # diagnostic counter, don't crash
    elif val_m.loss < best_val_loss:
        best_val_loss = val_m.loss
        best_val_acc  = val_m.accuracy
        torch.save(model.state_dict(), ckpt_path)

if nan_epochs:
    print(f"  WARN subject {sid:02d} fold {fold_idx}: "
          f"{nan_epochs}/{cfg.training.epochs} epochs had non-finite val loss")
    mlflow.log_metric(f"s{sid:02d}/fold{fold_idx}/nan_epochs", nan_epochs)
```

Effects:
- File always exists (worst case: random-init weights). MLflow can always log.
- NaN frequency surfaces as a first-class MLflow metric → downstream we can flag folds whose contribution to the subject mean is garbage.
- Crash converted to a soft signal — training completes for other subjects/folds without aborting the whole run.

## What this signals

- **Defensive contracts at I/O boundaries.** If operation B *must* consume a file written by operation A, write the file first and only update on improvement. The naive "save best" pattern is silently NaN-vulnerable.
- **IEEE NaN semantics catch experienced people repeatedly.** `nan < x`, `nan == nan`, `nan > x` all return False. Any "find the best" loop initialised with `float("inf")` is a latent bug waiting for the first non-finite value.
- **MPS is newer than CUDA.** Reproducibility expectations should reflect that. "Tests pass on MPS" doesn't mean they pass with the *same seed* on a different MPS driver build — backend non-determinism is more permissive than CUDA's deterministic flags.
- **Crash messages lie about location.** `FileNotFoundError` at `log_artifact` was the wrong place to look. Real bugs in pipelines often surface 3–5 layers downstream of the cause.
