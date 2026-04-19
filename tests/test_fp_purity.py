from __future__ import annotations

import ast
from pathlib import Path

PURE_DIRS = ("features", "state", "law", "decision", "backtest", "inference")
FORBIDDEN_NAMES = {"open", "print"}
FORBIDDEN_ATTRS = {
    ("datetime", "now"),
    ("time", "time"),
    ("os", "environ"),
    ("random", "random"),
}


def test_pure_core_has_no_obvious_io_side_effects() -> None:
    for directory in PURE_DIRS:
        for path in Path("src", directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in FORBIDDEN_NAMES, path
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    assert (node.value.id, node.attr) not in FORBIDDEN_ATTRS, path
