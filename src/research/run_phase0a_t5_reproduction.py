"""Run Phase 0A T5 reproduction provenance check.

io: inspects repository-local T5 sources and writes a structured report to stdout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_FILES = (
    "src/research/run_ndx_sigma_output_transform_pilot.py",
    "src/research/run_ndx_sigma_output_refinement_pilot.py",
    "src/research/run_ndx_sigma_output_monotone_family_scan.py",
)

EXCLUDED_SEARCH_PATHS = {
    "src/research/run_phase0a_t5_reproduction.py",
    "docs/phase0b/02_t5_reproduction_note.md",
}

KEYWORDS = (
    "T5",
    "residual persistence",
    "_fit_t5",
    "_predict_t5",
    "M4",
    "R1",
    "output transform",
    "rank correction",
    "monotone family",
)

WINDOWS = {
    "Window_2017": ("2017-07-07", "2017-12-29"),
    "Window_2018": ("2018-07-06", "2018-12-28"),
    "Window_2020": ("2020-01-03", "2020-06-26"),
}


@dataclass(frozen=True, slots=True)
class OriginDetail:
    name: str
    status: str
    evidence: str


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except UnicodeDecodeError:
        return ""


def _search_keywords() -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {keyword: [] for keyword in KEYWORDS}
    roots = (REPO_ROOT / "src", REPO_ROOT / "docs", REPO_ROOT / "artifacts")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".md", ".json", ".jsonl", ".txt"}:
                continue
            rel = str(path.relative_to(REPO_ROOT))
            if rel in EXCLUDED_SEARCH_PATHS:
                continue
            text = _read_text(path)
            for keyword in KEYWORDS:
                if keyword in text:
                    matches[keyword].append(rel)
    return {key: sorted(set(value)) for key, value in matches.items() if value}


def _expected_file_status() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in EXPECTED_FILES:
        path = REPO_ROOT / rel
        out.append({"path": rel, "exists": path.exists()})
    return out


def _origin_details(matches: dict[str, list[str]]) -> list[OriginDetail]:
    expected_exists = [item for item in _expected_file_status() if item["exists"]]
    direct_t5_sources = [
        path
        for keyword in ("_fit_t5", "_predict_t5", "residual persistence", "output transform")
        for path in matches.get(keyword, [])
        if path.startswith("src/research/")
    ]
    return [
        OriginDetail(
            name="T5 base",
            status="UNRESOLVED_ORIGIN_DETAIL",
            evidence=(
                "No repository-local T5 source identifies whether the base is M4, R1, "
                "or other."
            ),
        ),
        OriginDetail(
            name="T5 correction structure",
            status="UNRESOLVED_ORIGIN_DETAIL",
            evidence=(
                "No _fit_t5/_predict_t5/residual-persistence implementation found "
                "in src/research."
            ),
        ),
        OriginDetail(
            name="T5 train window and output frequency",
            status="UNRESOLVED_ORIGIN_DETAIL",
            evidence=(
                "Only benchmark lock defines rolling 416 weeks, 53 week embargo, "
                "weekly Friday output; no T5-specific source confirms this."
            ),
        ),
        OriginDetail(
            name="T5 final sigma_t generation",
            status="UNRESOLVED_ORIGIN_DETAIL",
            evidence=(
                "Expected T5 pilot scripts missing; no final sigma_t generation function found."
                if not expected_exists and not direct_t5_sources
                else (
                    "Partial references found but no executable T5 generation function "
                    "was identified."
                )
            ),
        ),
    ]


def run() -> int:
    matches = _search_keywords()
    origin_details = _origin_details(matches)
    window_results = {
        name: {
            "window": {"start": start, "end": end},
            "mean_z": "FAILED_TO_RUN_MISSING_T5_SOURCE",
            "std_z": "FAILED_TO_RUN_MISSING_T5_SOURCE",
            "corr_next": "FAILED_TO_RUN_MISSING_T5_SOURCE",
            "rank_next": "FAILED_TO_RUN_MISSING_T5_SOURCE",
            "lag1_acf_z": "FAILED_TO_RUN_MISSING_T5_SOURCE",
            "sigma_blowup": "FAILED_TO_RUN_MISSING_T5_SOURCE",
            "pathology": "FAILED_TO_RUN_MISSING_T5_SOURCE",
            "crps": "FAILED_TO_RUN_MISSING_T5_SOURCE",
            "status": "FAILED_TO_RUN_MISSING_T5_SOURCE",
        }
        for name, (start, end) in WINDOWS.items()
    }
    payload = {
        "status": "FAILED_TO_RUN_MISSING_T5_SOURCE",
        "reason": "T5 cannot be reconstructed from repository-local source without guessing.",
        "expected_files": _expected_file_status(),
        "keyword_matches": matches,
        "origin_details": [asdict(detail) for detail in origin_details],
        "actual_call_path": [
            "src/research/run_phase0a_t5_reproduction.py",
            "repository-local provenance scan",
            "FAILED_TO_RUN_MISSING_T5_SOURCE",
        ],
        "window_results": window_results,
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
