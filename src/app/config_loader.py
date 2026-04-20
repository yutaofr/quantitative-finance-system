"""Shell-only configuration assembly and artifact path helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from config_types import FrozenConfig

DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_ARTIFACTS_ROOT = Path("artifacts")
SRD_VERSION: Literal["8.7"] = "8.7"


@dataclass(frozen=True, slots=True)
class AdapterSecrets:
    """io: Adapter credential bundle kept out of FrozenConfig."""

    fred_api_key: str
    cboe_token: str


def _load_yaml(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    return cast(dict[str, Any], yaml.safe_load(content))


def _merge_runtime_values(  # noqa: PLR0913
    *,
    base: Mapping[str, Any],
    state: Mapping[str, Any],
    law: Mapping[str, Any],
    decision: Mapping[str, Any],
    backtest: Mapping[str, Any],
    data: Mapping[str, Any],
    env: Mapping[str, str],
    overrides: Mapping[str, object],
) -> FrozenConfig:
    random_seed = int(cast(Any, overrides.get("random_seed", base["random_seed"])))
    timezone = str(overrides.get("timezone", env.get("TZ", base["timezone"])))
    strict_pit_start = date.fromisoformat(
        str(overrides.get("strict_pit_start", data["vintage_modes"]["strict_start"])),
    )
    return FrozenConfig(
        srd_version=SRD_VERSION,
        random_seed=random_seed,
        timezone=timezone,
        strict_pit_start=strict_pit_start,
        missing_rate_degraded=float(state["missing_rate_degraded"]),
        missing_rate_blocked=float(state["missing_rate_blocked"]),
        quantile_gap=float(law["quantile_gap"]),
        l2_alpha=float(law["l2_alpha"]),
        tail_mult=float(law["tail_mult"]),
        utility_lambda=float(decision.get("lambda", 1.2)),
        utility_kappa=float(decision.get("kappa", 0.8)),
        band=float(decision["band"]),
        score_min=float(decision["score_min"]),
        score_max=float(decision["score_max"]),
        block_lengths=tuple(int(value) for value in backtest["block_lengths"]),
        bootstrap_replications=int(backtest["bootstrap_replications"]),
    )


def load_frozen_config(
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> FrozenConfig:
    """io: Load yaml once, merge CLI > env > yaml, and freeze runtime config."""
    merged_env = {} if env is None else dict(env)
    merged_overrides = {} if overrides is None else dict(overrides)
    return _merge_runtime_values(
        base=_load_yaml(config_dir / "base.yaml"),
        state=_load_yaml(config_dir / "state.yaml"),
        law=_load_yaml(config_dir / "law.yaml"),
        decision=_load_yaml(config_dir / "decision.yaml"),
        backtest=_load_yaml(config_dir / "backtest.yaml"),
        data=_load_yaml(config_dir / "data.yaml"),
        env=merged_env,
        overrides=merged_overrides,
    )


def load_adapter_secrets(env: Mapping[str, str]) -> AdapterSecrets:
    """io: Extract adapter credentials without mixing them into FrozenConfig."""
    return AdapterSecrets(
        fred_api_key=env.get("FRED_API_KEY", ""),
        cboe_token=env.get("CBOE_TOKEN", ""),
    )


def _as_of_label(as_of: date | str) -> str:
    if isinstance(as_of, date):
        return as_of.isoformat()
    return as_of


def weekly_artifact_dir(
    as_of: date | str,
    *,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
) -> Path:
    """io: Return the normalized SRD/ADD weekly artifact directory for one as-of date."""
    return artifacts_root / "weekly" / f"as_of={_as_of_label(as_of)}"


def production_output_path(
    as_of: date | str,
    *,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
) -> Path:
    """io: Return the normalized production output path for one as-of date."""
    return weekly_artifact_dir(as_of, artifacts_root=artifacts_root) / "production_output.json"


def production_hash_path(
    as_of: date | str,
    *,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
) -> Path:
    """io: Return the normalized production hash path for one as-of date."""
    return (
        weekly_artifact_dir(as_of, artifacts_root=artifacts_root) / "production_output.json.sha256"
    )
