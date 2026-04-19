"""Validate AGENTS.md, CLAUDE.md, and GEMINI.md consistency."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """io: read agent rulebooks and exit on mismatch."""
    paths = [Path("AGENTS.md"), Path("CLAUDE.md"), Path("GEMINI.md")]
    digests = {_digest(path) for path in paths}
    if len(digests) != 1:
        raise SystemExit("AGENTS.md, CLAUDE.md, and GEMINI.md differ")


if __name__ == "__main__":
    main()
