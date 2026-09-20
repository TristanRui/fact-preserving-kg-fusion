from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from hierarchy_fusion.data.case_graph import CaseGraph


@dataclass(frozen=True, slots=True)
class RoleAwareSparseResidualConfig:
    """Frozen residual settings reported by the manuscript."""

    role: str = "ROOT_CAUSE"
    minimum_similarity: float = 0.0317785682709644
    allowed_candidate_types: frozenset[str] = frozenset(
        {
            "D_bge_high_structure_low",
            "safe_or_low_risk_control",
        }
    )


class RoleAwareSparseIndex:
    """Character n-gram TF-IDF index over one diagnostic role.

    The vocabulary is learned without labels. Each case is represented by the
    concatenation of its events in one diagnostic role. Individual event texts
    are also included in the IDF corpus so mechanism phrases remain
    discriminative in long, templated case records.
    """

    def __init__(
        self,
        case_event_texts: Mapping[str, Sequence[str]],
        *,
        ngram_range: tuple[int, int] = (2, 4),
        max_features: int = 60_000,
    ) -> None:
        self.case_ids = tuple(sorted(case_event_texts))
        document_texts = [
            "；".join(text for text in case_event_texts[case_id] if text)
            for case_id in self.case_ids
        ]
        event_texts = [
            text for case_id in self.case_ids for text in case_event_texts[case_id] if text
        ]
        if not event_texts:
            raise ValueError("role-aware sparse index requires non-empty event text")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        matrix = self.vectorizer.fit_transform(document_texts + event_texts)
        self.case_matrix = matrix[: len(self.case_ids)]
        self.case_position = {case_id: position for position, case_id in enumerate(self.case_ids)}

    @classmethod
    def from_case_graphs(
        cls,
        graphs: Iterable[CaseGraph],
        *,
        role: str,
        ngram_range: tuple[int, int] = (2, 4),
        max_features: int = 60_000,
        minimum_text_characters: int = 1,
    ) -> RoleAwareSparseIndex:
        return cls(
            {
                graph.case_id: tuple(
                    node.text
                    for node in graph.nodes
                    if node.role == role
                    and len("".join(node.text.split())) >= minimum_text_characters
                )
                for graph in graphs
            },
            ngram_range=ngram_range,
            max_features=max_features,
        )

    def similarity(self, left_case_id: str, right_case_id: str) -> float:
        left = self.case_matrix[self.case_position[left_case_id]]
        right = self.case_matrix[self.case_position[right_case_id]]
        return float(left.multiply(right).sum())

    def aggregate_vector(self, case_ids: Iterable[str]) -> sparse.csr_matrix:
        """Return an L2-normalized sum over immutable leaf-case vectors.

        Topic representations never refit the vocabulary or IDF. This keeps the
        residual evidence space fixed to the label-free leaf corpus and makes a
        topic score exactly reproducible from its covered cases.
        """

        positions = [self.case_position[case_id] for case_id in sorted(set(case_ids))]
        if not positions:
            raise ValueError("aggregate vector requires at least one case")
        vector = sparse.csr_matrix(self.case_matrix[positions].sum(axis=0))
        norm = float(np.sqrt(vector.multiply(vector).sum()))
        if norm > 0.0:
            vector = vector / norm
        return vector.tocsr()

    def aggregate_similarity(
        self,
        left_case_ids: Iterable[str],
        right_case_ids: Iterable[str],
    ) -> float:
        left = self.aggregate_vector(left_case_ids)
        right = self.aggregate_vector(right_case_ids)
        return float(left.multiply(right).sum())


def fuse_role_aware_sparse_residual(
    *,
    structural_backbone_prediction: bool,
    candidate_type: str,
    role_similarity: float,
    config: RoleAwareSparseResidualConfig | None = None,
) -> bool:
    """Preserve the safe backbone and admit only a typed semantic residual."""

    active = config or RoleAwareSparseResidualConfig()
    if structural_backbone_prediction:
        return True
    return (
        candidate_type in active.allowed_candidate_types
        and float(role_similarity) >= active.minimum_similarity
    )
