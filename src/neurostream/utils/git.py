"""Git metadata helpers for provenance tracking."""

import subprocess


def git_sha() -> str:
    """Return the current HEAD commit SHA, or ``"unknown"`` outside a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
