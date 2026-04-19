"""Command-line shell for the QQQ law engine."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from enum import IntEnum
import sys


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

    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--start", required=True)
    backtest.add_argument("--end", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--window", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--as-of", required=True)

    return parser


def _not_implemented(command: str) -> int:
    """io: stderr status for intentionally incomplete production commands."""
    sys.stderr.write(
        f"{command} is not implemented yet; refusing to emit placeholder output.",
    )
    sys.stderr.write("\n")
    return int(ExitCode.BLOCKED)


def run(argv: Sequence[str] | None = None) -> int:
    """io: argv/stderr CLI dispatch."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.health:
        return int(ExitCode.OK)
    if args.command in {"weekly", "backtest", "train", "verify"}:
        return _not_implemented(str(args.command))
    parser.print_help(sys.stderr)
    return int(ExitCode.ENV_ERROR)


def main() -> None:
    """io: process exit entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
