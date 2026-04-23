from __future__ import annotations

import ast
from pathlib import Path

PANEL_CORE_MODULES = {
    "backtest.cluster_block_bootstrap",
    "backtest.panel_acceptance",
    "backtest.panel_metrics",
    "features.panel_block_builder",
    "law.panel_quantiles",
}
PANEL_FILES = {
    "src/app/panel_runner.py",
    "src/backtest/cluster_block_bootstrap.py",
    "src/backtest/panel_acceptance.py",
    "src/backtest/panel_metrics.py",
    "src/data_contract/asset_registry.py",
    "src/data_contract/yahoo_client.py",
    "src/features/panel_block_builder.py",
    "src/law/panel_quantiles.py",
}
FORBIDDEN_PATH_MARKERS = ("production_output.json", "artifacts/training")


def test_v87_modules_do_not_import_panel_core_modules() -> None:
    for path in Path("src").rglob("*.py"):
        if str(path) in PANEL_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in PANEL_CORE_MODULES, path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in PANEL_CORE_MODULES, path


def test_panel_code_does_not_target_production_artifact_paths() -> None:
    for raw_path in PANEL_FILES:
        content = Path(raw_path).read_text(encoding="utf-8")
        assert all(marker not in content for marker in FORBIDDEN_PATH_MARKERS), raw_path
