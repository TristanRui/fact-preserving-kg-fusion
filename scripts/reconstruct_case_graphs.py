"""Reconstruct provenance-preserving case graphs from frozen state decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from casefusion.io import read_jsonl, write_json, write_jsonl
from casefusion.reconstruction.case_graphs import build_case_graphs


def read_assignments(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and "assignments" in payload:
        payload = payload["assignments"]
    if not isinstance(payload, dict):
        raise ValueError("case assignments must be a JSON object")
    return {str(document_id): str(case_id) for document_id, case_id in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-graphs", type=Path, required=True)
    parser.add_argument("--case-assignments", type=Path, required=True)
    parser.add_argument("--event-decisions", type=Path, required=True)
    parser.add_argument("--relation-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    artifacts = build_case_graphs(
        read_jsonl(args.local_graphs),
        read_assignments(args.case_assignments),
        read_jsonl(args.event_decisions),
        read_jsonl(args.relation_decisions),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "case_graphs",
        "invalid_case_graphs",
        "event_mapping",
        "event_provenance",
        "relation_provenance",
        "review_queue",
    ):
        write_jsonl(args.output_dir / f"{name}.jsonl", artifacts[name])
    write_json(
        args.output_dir / "validation.json",
        artifacts["validation_report"],
    )
    write_json(args.output_dir / "summary.json", artifacts["summary"])
    return 0 if artifacts["validation_report"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
