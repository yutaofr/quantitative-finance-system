"""Environment contract helpers."""

from __future__ import annotations

from pathlib import Path

REQUIRED_ENV_KEYS = frozenset(
    {
        "FRED_API_KEY",
        "CBOE_TOKEN",
        "TZ",
        "LOG_LEVEL",
        "RUN_MODE",
        "AS_OF",
        "IMAGE_TAG",
        "DEPLOY_ENV",
    }
)


def read_env_keys(path: Path) -> frozenset[str]:
    """io: read dotenv-style keys from a local file."""
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, _value = stripped.partition("=")
        if separator:
            keys.add(key)
    return frozenset(keys)
