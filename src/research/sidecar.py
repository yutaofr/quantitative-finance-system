"""Research sidecar shell."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import orjson


def run(argv: Sequence[str] | None = None) -> int:
    """io: write research report placeholder only."""
    parser = argparse.ArgumentParser(prog="research-sidecar")
    parser.add_argument("--as-of", default="auto")
    args = parser.parse_args(argv)
    output_path = Path("artifacts") / "research_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": args.as_of,
        "srd_version": "8.7.1",
        "status": "not_implemented",
    }
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))
    return 0


def main() -> None:
    """io: process exit entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
