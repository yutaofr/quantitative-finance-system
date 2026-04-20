from __future__ import annotations

from pathlib import Path

from app.config_loader import (
    AdapterSecrets,
    load_adapter_secrets,
    load_frozen_config,
    production_output_path,
)


def test_load_frozen_config_merges_cli_over_env_over_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text(
        'srd_version: "8.7"\nrandom_seed: 8675309\ntimezone: America/New_York\n',
        encoding="utf-8",
    )
    (config_dir / "state.yaml").write_text(
        "missing_rate_degraded: 0.10\nmissing_rate_blocked: 0.20\n",
        encoding="utf-8",
    )
    (config_dir / "law.yaml").write_text(
        "quantile_gap: 1.0e-4\nl2_alpha: 2.0\ntail_mult: 0.6\n",
        encoding="utf-8",
    )
    (config_dir / "decision.yaml").write_text(
        "band: 7\nscore_min: 0\nscore_max: 100\n",
        encoding="utf-8",
    )
    (config_dir / "backtest.yaml").write_text(
        "block_lengths: [52, 78]\nbootstrap_replications: 2000\n",
        encoding="utf-8",
    )
    (config_dir / "data.yaml").write_text(
        'vintage_modes:\n  strict_start: "2012-01-06"\n',
        encoding="utf-8",
    )

    cfg = load_frozen_config(
        config_dir=config_dir,
        env={"TZ": "Europe/Paris"},
        overrides={"random_seed": 123},
    )

    assert cfg.timezone == "Europe/Paris"
    assert cfg.random_seed == 123
    assert cfg.band == 7.0


def test_load_adapter_secrets_keeps_credentials_out_of_frozen_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text(
        'srd_version: "8.7"\nrandom_seed: 8675309\ntimezone: America/New_York\n',
        encoding="utf-8",
    )
    (config_dir / "state.yaml").write_text(
        "missing_rate_degraded: 0.10\nmissing_rate_blocked: 0.20\n",
        encoding="utf-8",
    )
    (config_dir / "law.yaml").write_text(
        "quantile_gap: 1.0e-4\nl2_alpha: 2.0\ntail_mult: 0.6\n",
        encoding="utf-8",
    )
    (config_dir / "decision.yaml").write_text(
        "band: 7\nscore_min: 0\nscore_max: 100\n",
        encoding="utf-8",
    )
    (config_dir / "backtest.yaml").write_text(
        "block_lengths: [52, 78]\nbootstrap_replications: 2000\n",
        encoding="utf-8",
    )
    (config_dir / "data.yaml").write_text(
        'vintage_modes:\n  strict_start: "2012-01-06"\n',
        encoding="utf-8",
    )

    cfg = load_frozen_config(
        config_dir=config_dir,
        env={
            "FRED_API_KEY": "fred-secret",
            "CBOE_TOKEN": "cboe-secret",
        },
    )
    secrets = load_adapter_secrets(
        {
            "FRED_API_KEY": "fred-secret",
            "CBOE_TOKEN": "cboe-secret",
        },
    )

    assert isinstance(secrets, AdapterSecrets)
    assert not hasattr(cfg, "FRED_API_KEY")
    assert not hasattr(secrets, "nasdaq_dl_api_key")
    assert secrets.fred_api_key == "fred-secret"


def test_production_output_path_uses_as_of_partition(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    target = production_output_path("2024-12-27", artifacts_root=artifacts_root)
    assert target == artifacts_root / "weekly" / "as_of=2024-12-27" / "production_output.json"
