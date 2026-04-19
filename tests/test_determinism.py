from __future__ import annotations

from pathlib import Path

import pytest

from research.sidecar import run


def test_research_sidecar_placeholder_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert run(["--as-of", "2024-12-27"]) == 0
    first = Path("artifacts/research_report.json").read_bytes()
    assert run(["--as-of", "2024-12-27"]) == 0
    second = Path("artifacts/research_report.json").read_bytes()
    assert first == second
