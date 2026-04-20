from __future__ import annotations

from pathlib import Path

from app.cli import ExitCode, run
from app.config_loader import AdapterSecrets
from config_types import FrozenConfig

TOKEN_VALUE = "test" + "-api-token"


def test_health_returns_ok() -> None:
    assert run(["--health"]) == int(ExitCode.OK)


def _config() -> FrozenConfig:
    return FrozenConfig(
        srd_version="8.7",
        random_seed=8675309,
        timezone="America/New_York",
        missing_rate_degraded=0.10,
        missing_rate_blocked=0.20,
        quantile_gap=1.0e-4,
        l2_alpha=2.0,
        tail_mult=0.6,
        utility_lambda=1.2,
        utility_kappa=0.8,
        band=7.0,
        score_min=0.0,
        score_max=100.0,
        block_lengths=(52, 78),
        bootstrap_replications=2000,
    )


def test_weekly_uses_normalized_artifact_path_and_config_once(tmp_path: Path) -> None:
    seen: dict[str, object] = {}
    artifacts_root = tmp_path / "artifacts"

    def load_config(
        env: dict[str, str],
        overrides: dict[str, object] | None,
    ) -> FrozenConfig:
        seen["env"] = dict(env)
        seen["overrides"] = dict(overrides or {})
        return _config()

    def run_weekly_job(
        *,
        as_of: object,
        vintage_mode: str,
        cfg: FrozenConfig,
        output_path: Path,
        deps: object,
    ) -> int:
        seen["as_of"] = as_of
        seen["vintage_mode"] = vintage_mode
        seen["cfg"] = cfg
        seen["output_path"] = output_path
        seen["deps"] = deps
        return int(ExitCode.OK)

    assert run(
        ["weekly", "--as-of", "2024-12-27", "--artifacts-root", str(artifacts_root)],
        deps_overrides={
            "load_config": load_config,
            "run_weekly_job": run_weekly_job,
            "weekly_runner_deps": object(),
        },
        environ={"TZ": "America/New_York"},
    ) == int(ExitCode.OK)
    assert seen["vintage_mode"] == "strict"
    assert seen["cfg"] == _config()
    assert seen["output_path"] == (
        artifacts_root / "weekly" / "as_of=2024-12-27" / "production_output.json"
    )


def test_weekly_builds_default_runner_deps_without_storing_secrets_in_config(
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def load_config(
        env: dict[str, str],
        overrides: dict[str, object] | None,
    ) -> FrozenConfig:
        seen["config_env"] = dict(env)
        seen["overrides"] = dict(overrides or {})
        return _config()

    def load_secrets(env: dict[str, str]) -> AdapterSecrets:
        return AdapterSecrets(
            fred_api_key=env["FRED_API_KEY"],
            nasdaq_dl_api_key="",
            cboe_token="",
        )

    def build_deps(secrets: AdapterSecrets) -> object:
        seen["fred_secret"] = secrets.fred_api_key
        return object()

    def run_weekly_job(
        *,
        as_of: object,
        vintage_mode: str,
        cfg: FrozenConfig,
        output_path: Path,
        deps: object,
    ) -> int:
        del as_of, vintage_mode, output_path
        seen["cfg_has_secret"] = hasattr(cfg, "fred_api_key")
        seen["deps"] = deps
        return int(ExitCode.OK)

    assert run(
        ["weekly", "--as-of", "2024-12-27", "--artifacts-root", str(tmp_path)],
        deps_overrides={
            "load_config": load_config,
            "load_adapter_secrets": load_secrets,
            "build_weekly_runner_deps": build_deps,
            "run_weekly_job": run_weekly_job,
        },
        environ={"TZ": "America/New_York", "FRED_API_KEY": TOKEN_VALUE},
    ) == int(ExitCode.OK)
    assert seen["fred_secret"] == TOKEN_VALUE
    assert seen["cfg_has_secret"] is False


def test_verify_uses_same_artifact_directory_convention(tmp_path: Path) -> None:
    seen: dict[str, Path] = {}
    artifacts_root = tmp_path / "artifacts"

    def load_config(
        _env: dict[str, str],
        _overrides: dict[str, object] | None,
    ) -> FrozenConfig:
        return _config()

    def verify_artifact(*, as_of: object, output_path: Path, cfg: FrozenConfig) -> int:
        del as_of, cfg
        seen["output_path"] = output_path
        return int(ExitCode.HASH_MISMATCH)

    assert run(
        ["verify", "--as-of", "2024-12-27", "--artifacts-root", str(artifacts_root)],
        deps_overrides={
            "load_config": load_config,
            "verify_artifact": verify_artifact,
        },
        environ={"TZ": "America/New_York"},
    ) == int(ExitCode.HASH_MISMATCH)
    assert seen["output_path"] == (
        artifacts_root / "weekly" / "as_of=2024-12-27" / "production_output.json"
    )
