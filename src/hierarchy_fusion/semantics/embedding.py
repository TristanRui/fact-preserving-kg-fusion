from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from hierarchy_fusion.data.case_graph import CaseGraph

_BOUNDARY = re.compile(r"[。！？；\n]+")
_COMMA = re.compile(r"[，,]+")


def _split_evidence_units(text: str, maximum_characters: int = 96) -> list[str]:
    sentences = [part.strip(" ，,") for part in _BOUNDARY.split(text) if part.strip(" ，,")]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= maximum_characters:
            units.append(sentence)
            continue
        clauses = [part.strip() for part in _COMMA.split(sentence) if part.strip()]
        current = ""
        for clause in clauses:
            candidate = clause if not current else f"{current}，{clause}"
            if current and len(candidate) > maximum_characters:
                units.append(current)
                current = clause
            else:
                current = candidate
        if current:
            units.append(current)
    return units or [text.strip()]


def _unit_spans(text: str, units: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for unit in units:
        start = text.find(unit, cursor)
        if start < 0:
            start = text.find(unit)
        if start < 0:
            # Recombined long clauses can differ only in comma glyph. Locate
            # their first clause and use the full unit length conservatively.
            first_clause = _COMMA.split(unit)[0]
            start = text.find(first_clause, cursor)
        if start < 0:
            raise ValueError(f"cannot align evidence unit to source text: {unit[:40]}")
        end = min(len(text), start + len(unit))
        spans.append((start, end))
        cursor = end
    return spans


def load_case_bge_embeddings(
    graphs: list[CaseGraph],
    cache_path: Path,
    manifest_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path is not None
        else {"source": str(cache_path)}
    )
    cached = np.load(cache_path, allow_pickle=False)
    document_ids = [str(value) for value in cached["document_ids"]]
    matrix = np.asarray(cached["embeddings"], dtype=np.float32)
    if len(document_ids) != len(matrix):
        raise ValueError("BGE cache document ids and vectors are not aligned")
    by_document = {doc_id: matrix[idx] for idx, doc_id in enumerate(document_ids)}
    required = {doc_id for graph in graphs for doc_id in graph.document_ids}
    missing = sorted(required - set(by_document))
    if missing:
        raise ValueError(f"BGE cache misses documents: {missing[:10]}")
    by_case: dict[str, np.ndarray] = {}
    for graph in graphs:
        vector = np.mean([by_document[doc_id] for doc_id in graph.document_ids], axis=0)
        norm = float(np.linalg.norm(vector))
        by_case[graph.case_id] = (vector / max(norm, 1e-12)).astype(np.float32)
    return by_case, by_document, manifest


def load_event_context_bge_embeddings(
    graphs: list[CaseGraph],
    source_graph_path: Path,
    evidence_cache_path: Path,
    *,
    maximum_characters: int = 96,
) -> tuple[dict[str, np.ndarray], dict]:
    """Map each event to its containing frozen BGE evidence-unit vector."""
    rows = [
        json.loads(line)
        for line in source_graph_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    row_by_document = {str(row["document_id"]): row for row in rows}
    cached = np.load(evidence_cache_path, allow_pickle=False)
    document_ids = [str(value) for value in cached["document_ids"]]
    offsets = np.asarray(cached["unit_offsets"], dtype=np.int64)
    embeddings = np.asarray(cached["unit_embeddings"], dtype=np.float32)
    index_by_document = {doc_id: idx for idx, doc_id in enumerate(document_ids)}

    node_vectors: dict[str, np.ndarray] = {}
    unaligned = 0
    for graph in graphs:
        for node in graph.nodes:
            doc_idx = index_by_document[node.document_id]
            row = row_by_document[node.document_id]
            text = str(row["text"])
            units = _split_evidence_units(text, maximum_characters)
            start_idx, end_idx = int(offsets[doc_idx]), int(offsets[doc_idx + 1])
            if len(units) != end_idx - start_idx:
                raise ValueError(
                    f"evidence unit cache mismatch for {node.document_id}: "
                    f"source={len(units)} cache={end_idx - start_idx}"
                )
            spans = _unit_spans(text, units)
            overlaps = [max(0, min(node.end, end) - max(node.start, start)) for start, end in spans]
            if max(overlaps, default=0) <= 0:
                unaligned += 1
                unit_idx = min(range(len(spans)), key=lambda idx: abs(spans[idx][0] - node.start))
            else:
                unit_idx = int(np.argmax(overlaps))
            node_vectors[node.uid] = embeddings[start_idx + unit_idx]
    return node_vectors, {
        "node_count": len(node_vectors),
        "unaligned_node_count": unaligned,
        "embedding_dimension": int(embeddings.shape[1]),
        "source": str(evidence_cache_path),
        "pooling": "normalized_cls_evidence_unit",
    }
