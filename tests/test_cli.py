from __future__ import annotations

from app.cli import ExitCode, run


def test_health_returns_ok() -> None:
    assert run(["--health"]) == int(ExitCode.OK)


def test_weekly_refuses_placeholder_output() -> None:
    assert run(["weekly", "--as-of", "2024-12-27"]) == int(ExitCode.BLOCKED)
