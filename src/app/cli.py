"""Command-line shell for the QQQ law engine."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from enum import IntEnum
import hashlib
import os
from pathlib import Path
import sys
from typing import Any, cast

from app.backtest_runner import run_backtest_job
from app.config_loader import (
    load_adapter_secrets,
    load_frozen_config,
    production_hash_path,
    production_output_path,
)
from app.runtime_deps import (
    build_backtest_runner_deps,
    build_train_runner_deps,
    build_weekly_runner_deps,
)
from app.train_runner import parse_window_weeks, run_train_job
from app.weekly_runner import run_weekly_job
from config_types import FrozenConfig

LoadConfig = Callable[[dict[str, str], dict[str, object] | None], FrozenConfig]
RunWeekly = Callable[..., int]
VerifyArtifact = Callable[..., int]
LoadSecrets = Callable[[dict[str, str]], object]
BuildWeeklyRunnerDeps = Callable[..., object]
BuildTrainRunnerDeps = Callable[..., object]
BuildBacktestRunnerDeps = Callable[..., object]
RunTrain = Callable[..., int]
RunBacktest = Callable[..., int]


class ExitCode(IntEnum):
    """Documented process exit codes from ADD section 10."""

    OK = 0
    BLOCKED = 2
    ACCEPTANCE_FAILED = 3
    ENV_ERROR = 4
    HASH_MISMATCH = 5


def build_parser() -> argparse.ArgumentParser:
    """io: argv parser construction."""
    parser = argparse.ArgumentParser(prog="qqq-law-engine")
    parser.add_argument("--health", action="store_true", help="validate that the CLI imports")

    subparsers = parser.add_subparsers(dest="command")

    weekly = subparsers.add_parser("weekly")
    weekly.add_argument("--as-of", required=False, default="auto")
    weekly.add_argument("--artifacts-root", required=False, default="artifacts")

    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--start", required=True)
    backtest.add_argument("--end", required=True)
    backtest.add_argument("--artifacts-root", required=False, default="artifacts")

    train = subparsers.add_parser("train")
    train.add_argument("--window", required=True)
    train.add_argument("--as-of", required=False, default="auto")
    train.add_argument("--artifacts-root", required=False, default="artifacts")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--as-of", required=True)
    verify.add_argument("--artifacts-root", required=False, default="artifacts")

    return parser


def _not_implemented(command: str) -> int:
    """io: stderr status for intentionally incomplete production commands."""
    sys.stderr.write(
        f"{command} is not implemented yet; refusing to emit placeholder output.",
    )
    sys.stderr.write("\n")
    return int(ExitCode.BLOCKED)


def _parse_as_of(raw: str) -> date:
    if raw == "auto":
        return datetime.now(UTC).date()
    return date.fromisoformat(raw)


def _verify_artifact(
    *,
    as_of: date,
    output_path: Path,
    cfg: Any,
) -> int:
    del cfg
    hash_path = production_hash_path(as_of, artifacts_root=output_path.parents[2])
    if not output_path.exists() or not hash_path.exists():
        return int(ExitCode.BLOCKED)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    expected = hash_path.read_text(encoding="utf-8").strip()
    if digest == expected:
        return int(ExitCode.OK)
    return int(ExitCode.HASH_MISMATCH)


def run(
    argv: Sequence[str] | None = None,
    *,
    deps_overrides: dict[str, object] | None = None,
    environ: dict[str, str] | None = None,
) -> int:
    """io: argv/stderr CLI dispatch."""
    parser = build_parser()
    args = parser.parse_args(argv)
    env = dict(os.environ if environ is None else environ)
    deps = {} if deps_overrides is None else dict(deps_overrides)
    if args.health:
        return int(ExitCode.OK)
    if args.command == "weekly":
        load_config = cast(LoadConfig, deps.get("load_config", load_frozen_config))
        run_weekly = cast(RunWeekly, deps.get("run_weekly_job", run_weekly_job))
        load_secrets = cast(LoadSecrets, deps.get("load_adapter_secrets", load_adapter_secrets))
        build_deps = cast(
            BuildWeeklyRunnerDeps,
            deps.get("build_weekly_runner_deps", build_weekly_runner_deps),
        )
        weekly_runner_deps = deps.get("weekly_runner_deps")
        if weekly_runner_deps is None:
            weekly_runner_deps = build_deps(load_secrets(env))
        as_of = _parse_as_of(str(args.as_of))
        cfg = load_config(env, {})
        vintage_mode = "strict" if as_of >= cfg.strict_pit_start else "pseudo"
        output_path = production_output_path(as_of, artifacts_root=Path(args.artifacts_root))
        return run_weekly(
            as_of=as_of,
            vintage_mode=vintage_mode,
            cfg=cfg,
            output_path=output_path,
            deps=weekly_runner_deps,
        )
    if args.command == "verify":
        load_config = cast(LoadConfig, deps.get("load_config", load_frozen_config))
        verify_artifact = cast(VerifyArtifact, deps.get("verify_artifact", _verify_artifact))
        as_of = _parse_as_of(str(args.as_of))
        cfg = load_config(env, {})
        output_path = production_output_path(as_of, artifacts_root=Path(args.artifacts_root))
        return verify_artifact(as_of=as_of, output_path=output_path, cfg=cfg)
    if args.command == "train":
        load_config = cast(LoadConfig, deps.get("load_config", load_frozen_config))
        train_job = cast(RunTrain, deps.get("run_train_job", run_train_job))
        load_secrets = cast(LoadSecrets, deps.get("load_adapter_secrets", load_adapter_secrets))
        build_deps = cast(
            BuildTrainRunnerDeps,
            deps.get("build_train_runner_deps", build_train_runner_deps),
        )
        train_runner_deps = deps.get("train_runner_deps")
        if train_runner_deps is None:
            train_runner_deps = build_deps(load_secrets(env))
        as_of = _parse_as_of(str(args.as_of))
        cfg = load_config(env, {})
        vintage_mode = "strict" if as_of >= cfg.strict_pit_start else "pseudo"
        return train_job(
            as_of=as_of,
            vintage_mode=vintage_mode,
            cfg=cfg,
            training_root=Path(args.artifacts_root) / "training",
            window_weeks=parse_window_weeks(str(args.window)),
            deps=train_runner_deps,
        )
    if args.command == "backtest":
        load_config = cast(LoadConfig, deps.get("load_config", load_frozen_config))
        backtest_job = cast(RunBacktest, deps.get("run_backtest_job", run_backtest_job))
        load_secrets = cast(LoadSecrets, deps.get("load_adapter_secrets", load_adapter_secrets))
        build_deps = cast(
            BuildBacktestRunnerDeps,
            deps.get("build_backtest_runner_deps", build_backtest_runner_deps),
        )
        backtest_runner_deps = deps.get("backtest_runner_deps")
        if backtest_runner_deps is None:
            backtest_runner_deps = build_deps(load_secrets(env))
        start = _parse_as_of(str(args.start))
        end = _parse_as_of(str(args.end))
        cfg = load_config(env, {})
        return backtest_job(
            start=start,
            end=end,
            cfg=cfg,
            output_path=Path(args.artifacts_root) / "backtest" / "backtest_results.jsonl",
            deps=backtest_runner_deps,
        )
    parser.print_help(sys.stderr)
    return int(ExitCode.ENV_ERROR)


def main() -> None:
    """io: process exit entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
