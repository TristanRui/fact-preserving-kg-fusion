from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from hierarchy_fusion.coarsening.candidate_graph import build_ann_candidate_pairs
from hierarchy_fusion.coarsening.similarity import FixedLeafEvidenceIndex, association_score
from hierarchy_fusion.patterns.extractor import PatternInstance, pattern_identifier


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), 1e-12)


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    alpha: float
    beta: float
    tau_assoc: float
    candidate_top_k: int = 12
    max_merges: int = 357
    allow_multiple_roots: bool = True

    def validate(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("association weights must be non-negative")
        if abs(self.alpha + self.beta - 1.0) > 1e-9:
            raise ValueError("alpha and beta must sum to one for a stable score scale")
        if not 0.0 <= self.tau_assoc <= 1.0:
            raise ValueError("tau_assoc must be within [0, 1]")
        if self.candidate_top_k < 1:
            raise ValueError("candidate_top_k must be positive")
        if self.max_merges < 0:
            raise ValueError("max_merges must be non-negative")
        if not self.allow_multiple_roots:
            raise ValueError("the manuscript-aligned method must allow multiple roots")


@dataclass(slots=True)
class DiagnosticUnit:
    unit_id: str
    unit_type: str
    case_ids: frozenset[str]
    child_ids: tuple[str, ...]
    level: int
    vector: np.ndarray
    leaf_vector_sum: np.ndarray
    concept_ids: frozenset[str]
    evidence_features: frozenset[str]
    pattern_refs: frozenset[str]
    pattern_support_case_ids: dict[str, tuple[str, ...]]
    active: bool = True
    parent_topic_id: str | None = None

    @property
    def cluster_id(self) -> str:
        return self.unit_id

    @property
    def shared_patterns(self) -> frozenset[str]:
        return frozenset(
            signature
            for signature, case_ids in self.pattern_support_case_ids.items()
            if len(case_ids) >= 2
        )

    @property
    def pattern_support(self) -> tuple[dict, ...]:
        denominator = max(1, len(self.case_ids))
        return tuple(
            {
                "pattern_id": pattern_identifier(signature),
                "signature": signature,
                "supporting_case_ids": list(case_ids),
                "case_support_count": len(case_ids),
                "case_support_ratio": len(case_ids) / denominator,
                "evidence_admissible": bool(case_ids),
                "shared_pattern": len(case_ids) >= 2,
            }
            for signature, case_ids in sorted(self.pattern_support_case_ids.items())
        )

    def to_dict(self, *, include_vector: bool = False) -> dict:
        row = {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "child_unit_ids": list(self.child_ids),
            "covered_case_ids": sorted(self.case_ids),
            "level": self.level,
            "canonical_concept_ids": sorted(self.concept_ids),
            "evidence_features": sorted(self.evidence_features),
            "pattern_refs": [pattern_identifier(value) for value in sorted(self.pattern_refs)],
            "pattern_signatures": sorted(self.pattern_refs),
            "pattern_support_case_ids": {
                pattern_identifier(signature): list(case_ids)
                for signature, case_ids in sorted(self.pattern_support_case_ids.items())
            },
            "active_state": "active" if self.active else "inactive",
            "parent_topic_id": self.parent_topic_id,
        }
        if include_vector:
            row["semantic_vector"] = self.vector.tolist()
        return row


@dataclass(slots=True)
class HierarchyResult:
    variant: str
    units: dict[str, DiagnosticUnit]
    root_ids: tuple[str, ...]
    association_trace: list[dict]
    candidate_pair_count: int
    accepted_merge_count: int
    termination_reason: str
    config: AssociationConfig

    @property
    def clusters(self) -> dict[str, DiagnosticUnit]:
        """Return all case and topic units keyed by identifier."""
        return self.units

    @property
    def merge_trace(self) -> list[dict]:
        return self.association_trace

    def summary(self) -> dict:
        roots = [self.units[root_id] for root_id in self.root_ids]
        return {
            "variant": self.variant,
            "root_count": len(roots),
            "accepted_merge_count": self.accepted_merge_count,
            "candidate_pair_count": self.candidate_pair_count,
            "evaluated_candidate_count": len(self.association_trace),
            "termination_reason": self.termination_reason,
            "mean_cases_per_root": sum(len(root.case_ids) for root in roots) / max(1, len(roots)),
            "max_cases_per_root": max((len(root.case_ids) for root in roots), default=0),
            "max_level": max((root.level for root in roots), default=0),
        }


class HierarchyBuilder:
    """Recursive evidence-preserving topic formation from immutable case leaves."""

    def __init__(
        self,
        *,
        case_vectors: dict[str, np.ndarray],
        case_concepts: dict[str, set[str]],
        patterns_by_case: dict[str, list[PatternInstance]],
        evidence_index: FixedLeafEvidenceIndex,
        config: AssociationConfig,
    ) -> None:
        config.validate()
        if set(case_vectors) != set(case_concepts) or set(case_vectors) != set(patterns_by_case):
            raise ValueError("case vectors, concepts and patterns must cover the same leaves")
        self.case_vectors = {case_id: normalize(vector) for case_id, vector in case_vectors.items()}
        self.case_concepts = case_concepts
        self.patterns_by_case = patterns_by_case
        self.evidence_index = evidence_index
        self.config = config
        self.case_pattern_signatures = {
            case_id: frozenset(pattern.signature for pattern in patterns)
            for case_id, patterns in patterns_by_case.items()
        }
        self.pattern_support_index: dict[str, set[str]] = {}
        for case_id, signatures in self.case_pattern_signatures.items():
            for signature in signatures:
                self.pattern_support_index.setdefault(signature, set()).add(case_id)

    def _pattern_support(
        self,
        signatures: frozenset[str],
        case_ids: frozenset[str],
    ) -> dict[str, tuple[str, ...]]:
        return {
            signature: tuple(sorted(self.pattern_support_index.get(signature, set()) & case_ids))
            for signature in sorted(signatures)
        }

    def _leaf_units(self) -> tuple[dict[str, DiagnosticUnit], dict[str, str]]:
        units: dict[str, DiagnosticUnit] = {}
        case_to_leaf: dict[str, str] = {}
        for index, case_id in enumerate(sorted(self.case_vectors)):
            unit_id = f"C{index:04d}"
            case_to_leaf[case_id] = unit_id
            patterns = self.case_pattern_signatures[case_id]
            vector = self.case_vectors[case_id]
            units[unit_id] = DiagnosticUnit(
                unit_id=unit_id,
                unit_type="case",
                case_ids=frozenset({case_id}),
                child_ids=(),
                level=0,
                vector=vector,
                leaf_vector_sum=np.array(vector, dtype=np.float64, copy=True),
                concept_ids=frozenset(self.case_concepts[case_id]),
                evidence_features=self.evidence_index.case_features[case_id],
                pattern_refs=patterns,
                pattern_support_case_ids=self._pattern_support(patterns, frozenset({case_id})),
            )
        return units, case_to_leaf

    def score_units(self, left: DiagnosticUnit, right: DiagnosticUnit) -> dict[str, float]:
        semantic = max(0.0, float(np.clip(np.dot(left.vector, right.vector), -1.0, 1.0)))
        evidence = self.evidence_index.weighted_jaccard(
            left.evidence_features,
            right.evidence_features,
        )
        score = association_score(
            semantic,
            evidence,
            alpha=self.config.alpha,
            beta=self.config.beta,
        )
        return {
            "S_sem": semantic,
            "S_evi": evidence,
            "S_assoc": score,
        }

    def _parent(
        self,
        parent_id: str,
        left: DiagnosticUnit,
        right: DiagnosticUnit,
    ) -> DiagnosticUnit:
        case_ids = left.case_ids | right.case_ids
        leaf_sum = np.sum(
            [self.case_vectors[case_id] for case_id in sorted(case_ids)],
            axis=0,
        )
        pattern_refs = left.pattern_refs | right.pattern_refs
        evidence_features = left.evidence_features | right.evidence_features
        expected_features = self.evidence_index.topic_features(case_ids)
        if evidence_features != expected_features:
            raise AssertionError("topic evidence aggregation diverged from covered leaf union")
        return DiagnosticUnit(
            unit_id=parent_id,
            unit_type="topic",
            case_ids=case_ids,
            child_ids=(left.unit_id, right.unit_id),
            level=max(left.level, right.level) + 1,
            vector=normalize(leaf_sum),
            leaf_vector_sum=leaf_sum,
            concept_ids=left.concept_ids | right.concept_ids,
            evidence_features=evidence_features,
            pattern_refs=pattern_refs,
            pattern_support_case_ids=self._pattern_support(pattern_refs, case_ids),
        )

    def _topic_neighbors(
        self,
        topic: DiagnosticUnit,
        units: dict[str, DiagnosticUnit],
    ) -> list[str]:
        scores = [
            (float(np.dot(topic.vector, unit.vector)), unit_id)
            for unit_id, unit in units.items()
            if unit.active and unit_id != topic.unit_id
        ]
        return [
            unit_id
            for _, unit_id in sorted(scores, key=lambda row: (-row[0], row[1]))[
                : self.config.candidate_top_k
            ]
        ]

    def build(
        self,
        *,
        variant: str = "Proposed_hierarchy_fusion",
        initial_candidate_pairs: set[tuple[str, str]] | None = None,
    ) -> HierarchyResult:
        units, case_to_leaf = self._leaf_units()
        case_ids = sorted(self.case_vectors)
        raw_pairs = initial_candidate_pairs
        if raw_pairs is None:
            raw_pairs = build_ann_candidate_pairs(
                case_ids,
                self.case_vectors,
                top_k=self.config.candidate_top_k,
            )
        heap: list[tuple[float, str, str, dict[str, float]]] = []
        queued: set[tuple[str, str]] = set()

        def push(left_id: str, right_id: str) -> None:
            if left_id == right_id:
                return
            left_id, right_id = sorted((left_id, right_id))
            pair = (left_id, right_id)
            if pair in queued:
                return
            left, right = units[left_id], units[right_id]
            values = self.score_units(left, right)
            heapq.heappush(heap, (-values["S_assoc"], left_id, right_id, values))
            queued.add(pair)

        for left_case, right_case in sorted(raw_pairs):
            push(case_to_leaf[left_case], case_to_leaf[right_case])

        trace: list[dict] = []
        accepted = 0
        next_topic = 0
        iteration = 0
        termination = "candidate_set_exhausted"
        while heap:
            _, left_id, right_id, scores = heapq.heappop(heap)
            left, right = units[left_id], units[right_id]
            if not left.active or not right.active:
                continue
            iteration += 1
            record = {
                "iteration": iteration,
                "unit_a": left_id,
                "unit_b": right_id,
                "unit_a_type": left.unit_type,
                "unit_b_type": right.unit_type,
                "covered_case_ids_a": sorted(left.case_ids),
                "covered_case_ids_b": sorted(right.case_ids),
                **scores,
                "tau_assoc": self.config.tau_assoc,
            }
            if scores["S_assoc"] < self.config.tau_assoc:
                record.update(
                    {
                        "accepted": False,
                        "parent_topic_id": None,
                        "reason": "below_association_threshold",
                    }
                )
                trace.append(record)
                termination = "best_candidate_below_tau_assoc"
                break
            if accepted >= self.config.max_merges:
                record.update(
                    {
                        "accepted": False,
                        "parent_topic_id": None,
                        "reason": "max_merges_reached",
                    }
                )
                trace.append(record)
                termination = "max_merges_reached"
                break

            parent_id = f"T{next_topic:04d}"
            next_topic += 1
            parent = self._parent(parent_id, left, right)
            units[parent_id] = parent
            left.active = False
            right.active = False
            left.parent_topic_id = parent_id
            right.parent_topic_id = parent_id
            accepted += 1
            record.update(
                {
                    "accepted": True,
                    "parent_topic_id": parent_id,
                    "reason": "accepted_by_association",
                }
            )
            trace.append(record)
            for other_id in self._topic_neighbors(parent, units):
                push(parent_id, other_id)

        roots = tuple(sorted(unit_id for unit_id, unit in units.items() if unit.active))
        return HierarchyResult(
            variant=variant,
            units=units,
            root_ids=roots,
            association_trace=trace,
            candidate_pair_count=len(queued),
            accepted_merge_count=accepted,
            termination_reason=termination,
            config=self.config,
        )
