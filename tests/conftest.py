# tests/conftest.py
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Root of the repo, where pyproject.toml lives."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def split_path(project_root: Path) -> Path:
    return project_root / "src" / "neurostream" / "data" / "bci_iv_2a_v1.json"
