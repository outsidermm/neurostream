# 07 — MLflow portability: container paths in `meta.yaml`

*MLflow's local file store bakes absolute paths into experiment metadata. Switching from container to host broke every run.*

## What we observed

- First training run on host macOS (after working in the dev container) crashed with:

```
OSError: [Errno 30] Read-only file system: '/workspaces'
```

- The path didn't exist on the host — `/workspaces` is the container's mount point. macOS has no such directory and no permission to create one at the filesystem root.

## What caused it

- The MLflow experiment was first created **inside the dev container**, where the working directory was `/workspaces/neurostream/`
- MLflow wrote the absolute container path into the experiment's `meta.yaml`:

```yaml
artifact_location: file:///workspaces/neurostream/mlruns/315551855380257610
```

- On the host, MLflow read that file, saw the container path, and tried to `mkdir -p /workspaces/...` — the OS refused
- The same `mlruns/` directory was shared between two filesystems that disagreed on absolute path layout

There was a deprecation warning from MLflow telegraphing exactly this class of failure:

```
FutureWarning: The filesystem tracking backend (e.g., './mlruns') is deprecated
as of February 2026. Consider transitioning to a database backend (e.g.,
'sqlite:///mlflow.db') ...
```

## What we did

Wiped the broken `mlruns/` directory on the host and re-ran:

```bash
rm -rf mlruns/
uv run python -m neurostream.training.train
```

New `mlruns/` is created with host paths; subsequent host runs work.

**Documented working rule:** run training in *one* environment, not both. Dev work happens in the container; training happens on the host (for MPS speedup) — but the two never share the same `mlruns/`.

Alternatives considered (deferred to Phase 4):

- **SQLite backend** (`mlflow.tracking_uri: sqlite:///mlflow.db`) — paths still get stored, just inside the DB. Same problem, different format.
- **Remote tracking server** — proper fix; not worth the operational cost at Phase 1.

## What this signals

- **Filesystem paths leak across environments.** Anything that records absolute paths into config files breaks the moment the directory layout changes. Container ↔ host is the most common variant; CI ↔ local is the next.
- **Read the deprecation warnings.** MLflow was telegraphing this issue weeks in advance. Treating `FutureWarning` as informational noise costs you a debugging hour later.
- **"It works in my container" is a real risk for ML projects** where MLflow / DVC / artifact stores write absolute paths invisibly. Same root cause as dependency pinning — the difference is the metadata is *generated*, not committed, so it's invisible until it fails.
- The fact that the fix was "delete and restart" is itself a signal that the local file store is unsuitable for any multi-environment workflow. A real production setup mandates a remote tracking server with content-addressed artifacts — exactly the Phase 4 deliverable.
