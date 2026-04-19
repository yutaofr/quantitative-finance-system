from __future__ import annotations

import ast
from pathlib import Path

PURE_DIRS = ("features", "state", "law", "decision", "backtest", "inference")
FORBIDDEN_PREFIXES = ("app", "data_contract", "research")


def _imported_root(alias: ast.alias) -> str:
    return alias.name.split(".", maxsplit=1)[0]


def test_pure_core_does_not_import_forbidden_layers() -> None:
    for directory in PURE_DIRS:
        for path in Path("src", directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = {_imported_root(alias) for alias in node.names}
                    assert imported.isdisjoint(FORBIDDEN_PREFIXES), path
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".", maxsplit=1)[0] not in FORBIDDEN_PREFIXES, path
