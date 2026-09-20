"""Encode record text with BAAI/bge-small-zh-v1.5 into an NPZ cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from casefusion.embeddings import encode_bge_texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    document_ids = np.asarray([str(row["document_id"]) for row in rows])
    embeddings = encode_bge_texts(
        [str(row["text"]) for row in rows],
        model_name=args.model,
        batch_size=args.batch_size,
        maximum_length=512,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        document_ids=document_ids,
        embeddings=embeddings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
