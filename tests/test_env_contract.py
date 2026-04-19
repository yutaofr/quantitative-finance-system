from __future__ import annotations

from pathlib import Path

from app.env_validator import REQUIRED_ENV_KEYS, read_env_keys


def test_env_example_declares_required_keys() -> None:
    assert read_env_keys(Path(".env.example")) == REQUIRED_ENV_KEYS
