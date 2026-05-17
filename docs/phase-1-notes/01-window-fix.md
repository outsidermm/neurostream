# 01 — Trial window fix: TMAX 4.0 → 3.9

*Cutting the per-trial window by 100 ms kept the `len(epochs) == 288` assertion true.*

## What we observed

- Loader assertion failed for session T: `assert len(epochs) == 288, f"Expected 288 trials, got {len(epochs)}"` reported 287
- MNE log noted one trial dropped with reason `TOO_SHORT`
- Failure was deterministic — same subject, same trial every run

## What caused it

- BCI IV 2a recordings end **exactly 5.908 s** after the final cue across every training session — an undocumented dataset quirk
- A 4.0 s window from cue onset for the last trial overshoots the recording's end
- MNE silently drops trials that don't fit the requested window; the 288 assertion is what surfaced the silence

## What we did

- File: [`src/neurostream/data/loader.py`](../../src/neurostream/data/loader.py)
- Set `TMIN, TMAX = 0, 3.9` — 0.092 s safety margin inside the 5.908 s post-cue buffer
- Left a code comment quoting the 5.908 s figure so the next reader doesn't bump it back to 4.0
- *Superseded later* by the paper-faithful `[0.5, 2.5]` s window — see [05-resampling-protocol.md](05-resampling-protocol.md)

```python
# loader.py — pre-resampling era
TMIN, TMAX = 0, 3.9   # 3.9 keeps the 288th trial; 4.0 drops it as TOO_SHORT
```

## What this signals

A loader that silently throws away one trial is the kind of bug that costs 0.4 pp of accuracy and ten hours of investigation. The fix isn't to weaken the assertion — it's to treat assertion failures as a *data-understanding* problem and find the dataset quirk responsible. Catching these early prevents whole categories of "why does my baseline drift across runs" mysteries downstream.
