# Environment and Setup

The constraints any execution step runs under. Read before running a script.

## Hardware and OS

- **Windows native**, PowerShell (not WSL or Git Bash unless explicitly switched).
- GPU: **RTX 5060, 8GB VRAM** — a hard constraint. It forces pretraining batch
  size to 64 (vs. the recipe's 256). Don't propose configs that assume more VRAM.
- PyTorch must be installed with the matching CUDA build or it silently falls
  back to CPU. Install with
  `pip install torch --index-url https://download.pytorch.org/whl/cu126` (match
  the cu1xx tag to the installed CUDA toolkit), then confirm with
  `torch.cuda.is_available()`.
- Package management: **`uv`**. Python deps go through `uv` per the Phase 1
  toolchain decision.

## PowerShell

- **Backslash `\` line continuation is Bash syntax and does not work.**
  PowerShell uses a backtick `` ` `` that must be the literal last character on
  the line (no trailing whitespace). Prefer **single-line commands with absolute
  paths** over multi-line continuation — more reliable in this environment.
- Correct form:
  ```powershell
  python -m scripts.linear_probe probe.pretrained_checkpoint=C:\full\absolute\path\to\ckpt.pt
  ```

## Hydra

- **Hydra changes the working directory by default**, so relative paths in CLI
  overrides resolve against Hydra's per-run output directory, not the launch
  directory — a silent `FileNotFoundError` source. Use **absolute paths** in CLI
  overrides (current approach), or set `hydra: job: chdir: false` to disable the
  directory change project-wide.

## MLflow

- **The MLflow server is not a background service — start it manually** before
  any script that logs to it, in its own terminal, left running:
  ```powershell
  mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
  ```
- **Start it from the project root** (where `mlflow.db` lives), or MLflow
  creates a fresh empty database and previous run history disappears from the UI
  (not deleted — just not shown). Check `ls mlflow.db` before launching if unsure.
- UI at `http://127.0.0.1:5000`. Convention: group runs that should be compared
  directly into one experiment; don't create a new experiment per config tweak.

## Checkpoint filenames

- Format: `milestone_step{N:08d}.pt` / `rolling_step{N:08d}.pt` — **8-digit
  zero-padded** step number. `milestone_step01200000.pt` is correct; any other
  digit count silently fails to match and raises `FileNotFoundError`.
- **Rolling checkpoints are pruned** — only the last K (default 5, per
  `CheckpointManager`) survive on disk. Only milestone checkpoints
  (200k/400k/600k/800k/1.2M) are permanent; a non-milestone early-training
  checkpoint cannot be recovered without re-running pretraining to that step.

## Console / logging

- **Non-ASCII glyphs in log messages can crash logging on Windows** — the
  default GBK console codec can't encode characters like `✗`, producing a
  `UnicodeEncodeError`. This happens in the logging emit step, *after* the result
  is already computed and saved (to MLflow, JSON, etc.) — it is cosmetic, not a
  failure. Keep log strings ASCII to avoid it.

## Pip in sandboxed environments

- In sandboxed/constrained environments (not the Windows dev machine),
  `pip install` may need `--break-system-packages`. Relevant only if testing this
  code in a different container/sandbox.
