"""Resolve maintenance records into predicted case identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from casefusion.case_resolution import CaseResolutionConfig, resolve_cases


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache = np.load(args.embeddings, allow_pickle=False)
    document_ids = [str(value) for value in cache["document_ids"]]
    matrix = np.asarray(cache["embeddings"], dtype=np.float32)
    embeddings = {document_id: matrix[index] for index, document_id in enumerate(document_ids)}
    result = resolve_cases(
        read_jsonl(args.records),
        embeddings,
        config=CaseResolutionConfig(),
    )
    payload = {
        "schema_version": "case_resolution_v1",
        "method": "bm25_bge_hard_conflict_average_linkage",
        "assignments": result.assignments,
        "candidate_decisions": [decision.to_dict() for decision in result.candidate_decisions],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
