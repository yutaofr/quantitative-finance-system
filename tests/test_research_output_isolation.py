from __future__ import annotations

from pathlib import Path

import pytest

from research.sidecar import run


def test_research_sidecar_writes_only_research_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert run(["--as-of", "2024-12-27"]) == 0
    files = sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert files == ["artifacts/research_report.json"]
