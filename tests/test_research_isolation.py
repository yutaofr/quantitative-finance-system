from __future__ import annotations

import ast
from pathlib import Path

PRODUCTION_DIRS = (
    "app",
    "backtest",
    "data_contract",
    "decision",
    "features",
    "inference",
    "law",
    "state",
)


def test_production_code_does_not_import_research() -> None:
    for directory in PRODUCTION_DIRS:
        for path in Path("src", directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(not alias.name.startswith("research") for alias in node.names), path
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("research"), path
