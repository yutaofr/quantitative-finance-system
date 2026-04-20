from __future__ import annotations

import pytest

from app.cli import run


@pytest.mark.integration
def test_cli_health_integration_smoke() -> None:
    assert run(["--health"]) == 0
