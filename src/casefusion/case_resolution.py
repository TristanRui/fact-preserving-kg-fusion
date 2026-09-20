"""Manuscript-aligned maintenance-record case resolution.

The production score is the frozen combination reported in Section 3.3:
0.75 symmetric BM25 + 0.25 BGE cosine similarity. High-confidence diagnostic
conflicts veto an edge, and average-linkage clustering converts pair scores
into case assignments.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from casefusion.extraction.slots import extract_corpus_slots
from casefusion.matching.conflicts import detect_conflicts
from casefusion.text import domain_tokens


@dataclass(frozen=True, slots=True)
class CaseResolutionConfig:
    bm25_weight: float = 0.75
    bge_weight: float = 0.25
    decision_threshold: float = 0.40817840781035897
    candidate_top_k: int = 20
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def validate(self) -> None:
        if not math.isclose(self.bm25_weight + self.bge_weight, 1.0, abs_tol=1e-12):
            raise ValueError("BM25 and BGE weights must sum to one")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be within [0, 1]")
        if self.candidate_top_k < 1:
            raise ValueError("candidate_top_k must be positive")


@dataclass(frozen=True, slots=True)
class CasePairDecision:
    document_id_a: str
    document_id_b: str
    bm25_similarity: float
    bge_similarity: float
    same_case_score: float
    hard_conflict: bool
    conflict_evidence: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["conflict_evidence"] = list(self.conflict_evidence)
        return value


@dataclass(frozen=True, slots=True)
class CaseResolutionResult:
    assignments: dict[str, str]
    candidate_decisions: tuple[CasePairDecision, ...]


class SymmetricBM25:
    """BM25 normalized by each query's self-score and averaged by direction."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.idf: dict[str, float] = {}
        self.average_document_length = 1.0

    def fit(self, texts: Sequence[str]) -> SymmetricBM25:
        tokenized = [domain_tokens(text) for text in texts]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        document_count = len(tokenized)
        self.idf = {
            token: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.average_document_length = sum(map(len, tokenized)) / max(document_count, 1)
        return self

    def _score(self, query: Sequence[str], document: Sequence[str]) -> float:
        if not query or not document:
            return 0.0
        frequencies = Counter(document)
        length_factor = 1.0 - self.b + self.b * len(document) / self.average_document_length
        score = 0.0
        for token in set(query):
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            score += self.idf.get(token, 0.0) * (
                frequency * (self.k1 + 1.0) / (frequency + self.k1 * length_factor)
            )
        return score

    def similarity(self, left: Sequence[str], right: Sequence[str]) -> float:
        left_self = self._score(left, left)
        right_self = self._score(right, right)
        left_to_right = self._score(left, right) / left_self if left_self else 0.0
        right_to_left = self._score(right, left) / right_self if right_self else 0.0
        return float(np.clip((left_to_right + right_to_left) / 2.0, 0.0, 1.0))


def _normalized_embeddings(
    document_ids: Sequence[str], embeddings: Mapping[str, np.ndarray]
) -> np.ndarray:
    missing = sorted(set(document_ids) - set(embeddings))
    if missing:
        raise ValueError(f"missing BGE embeddings for documents: {missing[:5]}")
    matrix = np.stack(
        [np.asarray(embeddings[document_id], dtype=np.float64) for document_id in document_ids]
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _candidate_pairs(similarities: np.ndarray, top_k: int) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for left_index, row in enumerate(similarities):
        ranking = np.argsort(-row, kind="stable")
        selected = 0
        for right_index_raw in ranking:
            right_index = int(right_index_raw)
            if left_index == right_index:
                continue
            pairs.add(tuple(sorted((left_index, right_index))))
            selected += 1
            if selected >= min(top_k, len(row) - 1):
                break
    return pairs


def _stable_assignments(document_ids: Sequence[str], labels: np.ndarray) -> dict[str, str]:
    groups: dict[int, list[str]] = {}
    for document_id, label in zip(document_ids, labels, strict=True):
        groups.setdefault(int(label), []).append(document_id)
    ordered = sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])
    return {
        document_id: f"CASE_{index:06d}"
        for index, group in enumerate(ordered, start=1)
        for document_id in group
    }


def resolve_cases(
    records: Sequence[Mapping[str, Any]],
    bge_embeddings: Mapping[str, np.ndarray],
    *,
    config: CaseResolutionConfig | None = None,
) -> CaseResolutionResult:
    """Resolve records into cases without reading any reference identity field."""

    active = config or CaseResolutionConfig()
    active.validate()
    sanitized = [
        {"document_id": str(record["document_id"]), "text": str(record["text"])}
        for record in records
    ]
    document_ids = [record["document_id"] for record in sanitized]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document_id values must be unique")
    if len(document_ids) < 2:
        assignments = {document_id: "CASE_000001" for document_id in document_ids}
        return CaseResolutionResult(assignments, ())

    texts = [record["text"] for record in sanitized]
    tokens = [domain_tokens(text) for text in texts]
    bm25 = SymmetricBM25(k1=active.bm25_k1, b=active.bm25_b).fit(texts)
    bge_matrix = _normalized_embeddings(document_ids, bge_embeddings)
    bge_scores = np.clip(bge_matrix @ bge_matrix.T, 0.0, 1.0)

    hybrid_scores = np.eye(len(document_ids), dtype=np.float64)
    for left_index in range(len(document_ids)):
        for right_index in range(left_index + 1, len(document_ids)):
            bm25_score = bm25.similarity(tokens[left_index], tokens[right_index])
            hybrid = (
                active.bm25_weight * bm25_score
                + active.bge_weight * bge_scores[left_index, right_index]
            )
            hybrid_scores[left_index, right_index] = hybrid_scores[right_index, left_index] = hybrid

    candidates = _candidate_pairs(hybrid_scores, active.candidate_top_k)
    slots = {row["document_id"]: row for row in extract_corpus_slots(sanitized)}
    sparse_scores = np.zeros_like(hybrid_scores)
    np.fill_diagonal(sparse_scores, 1.0)
    decisions: list[CasePairDecision] = []
    for left_index, right_index in sorted(candidates):
        left_id = document_ids[left_index]
        right_id = document_ids[right_index]
        conflict = detect_conflicts(slots[left_id], slots[right_id])
        bm25_score = bm25.similarity(tokens[left_index], tokens[right_index])
        bge_score = float(bge_scores[left_index, right_index])
        score = active.bm25_weight * bm25_score + active.bge_weight * bge_score
        if conflict["has_hard_conflict"]:
            score = 0.0
        sparse_scores[left_index, right_index] = sparse_scores[right_index, left_index] = score
        decisions.append(
            CasePairDecision(
                document_id_a=left_id,
                document_id_b=right_id,
                bm25_similarity=bm25_score,
                bge_similarity=bge_score,
                same_case_score=float(score),
                hard_conflict=bool(conflict["has_hard_conflict"]),
                conflict_evidence=tuple(conflict["conflicts"]),
            )
        )

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - active.decision_threshold,
    )
    labels = model.fit_predict(1.0 - sparse_scores)
    return CaseResolutionResult(
        assignments=_stable_assignments(document_ids, labels),
        candidate_decisions=tuple(decisions),
    )
