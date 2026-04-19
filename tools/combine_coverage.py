"""Combine coverage reports placeholder for early scaffold CI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """io: parse coverage gate arguments.

    The full coverage aggregation gate belongs with production implementation. During scaffold
    initialization this command validates that CI wiring can call the tool.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="*")
    parser.add_argument("--global-min", type=int, required=True)
    parser.add_argument("--domain-min", type=int, required=True)
    parser.add_argument("--domain-paths", nargs="+", required=True)
    parser.parse_args(argv)


if __name__ == "__main__":
    main()
