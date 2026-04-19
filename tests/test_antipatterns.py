from __future__ import annotations

import ast
from pathlib import Path


def test_tests_do_not_define_reference_or_mirror_algorithms() -> None:
    for path in Path("tests").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assert not node.name.startswith(("_reference_", "_mirror_", "_my_")), path
