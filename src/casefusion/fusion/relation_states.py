"""Explicit relation-state inference for conservative case-graph fusion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from casefusion.fusion.event_states import EventMappingSupport, mention_similarity
from casefusion.text import character_dice_similarity

RELATION_STATES = ("equivalent", "complementary", "conflicting", "unrelated", "uncertain")
STATE_TO_ACTION = {
    "equivalent": "merge",
    "complementary": "retain_parallel",
    "conflicting": "mark_conflict",
    "unrelated": "retain_separate",
    "uncertain": "retain_for_review",
}


@dataclass(frozen=True)
class RelationFeatureRecord:
    direct_head_text: float
    direct_tail_text: float
    reverse_head_text: float
    reverse_tail_text: float
    evidence_similarity: float
    relation_type_match: float
    relation_role_compatible: float
    modality_match: float
    head_mapping_support: float
    tail_mapping_support: float
    reverse_head_mapping_support: float
    reverse_tail_mapping_support: float
    head_forbidden_risk: float
    tail_forbidden_risk: float
    numeral_conflict: float
    negation_conflict: float
    maintenance_state_conflict: float
    reverse_direction_conflict: float
    bge_direct_head_text: float = 0.0
    bge_direct_tail_text: float = 0.0
    bge_reverse_head_text: float = 0.0
    bge_reverse_tail_text: float = 0.0
    bge_evidence_similarity: float = 0.0
    relation_semantic_similarity: float = 0.0
    action_semantic_similarity: float = 0.0
    action_semantic_available: float = 0.0
    narrative_position_similarity: float = 0.0

    @property
    def direct_endpoint_support(self) -> float:
        return min(self.direct_head_text, self.direct_tail_text)

    @property
    def mapping_endpoint_support(self) -> float:
        return min(self.head_mapping_support, self.tail_mapping_support)

    @property
    def shared_endpoint_support(self) -> float:
        return max(
            self.head_mapping_support,
            self.tail_mapping_support,
            self.reverse_head_mapping_support,
            self.reverse_tail_mapping_support,
        )

    @property
    def bge_direct_endpoint_support(self) -> float:
        return min(self.bge_direct_head_text, self.bge_direct_tail_text)

    @property
    def bge_reverse_endpoint_support(self) -> float:
        return min(self.bge_reverse_head_text, self.bge_reverse_tail_text)

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class RelationStateDecision:
    relation_a_id: str
    relation_b_id: str
    predicted_state: str
    graph_action: str
    equivalence_score: float
    complementary_probability: float | None
    confidence: float
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    endpoint_mapping_support: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    provenance: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reason_codes"] = list(self.reason_codes)
        result["provenance"] = list(self.provenance)
        return result


def endpoint_state(text: str) -> str:
    if re.search(r"(?:有无|是否|检查|复查|确认|验证|试验)", text):
        return "inspection"
    if re.search(r"(?:无法|不能|未采用|未安装|没有|为零|无压力|不合格|失效|仍然|又未)", text):
        return "negative"
    if re.search(r"(?:采用新|更换|修复|完成装配|合格|恢复正常|有效防松|已完成)", text):
        return "repaired"
    if re.search(r"(?:裂纹|泄漏|松动|磨损|断裂|变形|异响|碰磨|损伤|深槽)", text):
        return "failure"
    return "neutral"


def _numerals(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def numeral_conflict(left: str, right: str) -> bool:
    left_numbers = _numerals(left)
    right_numbers = _numerals(right)
    return bool(left_numbers and right_numbers and left_numbers != right_numbers)


def negation_conflict(left: str, right: str) -> bool:
    # Interrogative inspection phrases (e.g. “检查有无损伤”) are not factual
    # negations and must not conflict with a confirmed inspection target.
    left = re.sub(r"(?:检查|复查|确认|验证)?(?:有无|是否)", "", left)
    right = re.sub(r"(?:检查|复查|确认|验证)?(?:有无|是否)", "", right)
    pattern = re.compile(r"(?:无|未|没有|不得|不应|不能|消失|为零)")
    return bool(pattern.search(left)) != bool(pattern.search(right))


def maintenance_state_conflict(left: str, right: str) -> bool:
    states = {endpoint_state(left), endpoint_state(right)}
    return states == {"negative", "repaired"}


def relation_role_compatible(left: str, right: str) -> bool:
    left = left.upper()
    right = right.upper()
    compatible = {
        ("CAUSE", "CAUSE"),
        ("TREAT", "TREAT"),
        ("VERIFY", "VERIFY"),
        ("TEMPORAL", "TEMPORAL"),
        ("CAUSE", "TREAT"),
        ("TREAT", "CAUSE"),
        ("TREAT", "VERIFY"),
        ("VERIFY", "TREAT"),
        ("TEMPORAL", "TREAT"),
        ("TREAT", "TEMPORAL"),
        ("TEMPORAL", "VERIFY"),
        ("VERIFY", "TEMPORAL"),
        ("TEMPORAL", "CAUSE"),
        ("CAUSE", "TEMPORAL"),
        ("CAUSE", "VERIFY"),
        ("VERIFY", "CAUSE"),
    }
    return (left, right) in compatible


def relation_features(
    row: Mapping[str, Any],
    mappings: Mapping[str, EventMappingSupport] | None = None,
    semantic_context: Mapping[str, float] | None = None,
) -> RelationFeatureRecord:
    mappings = mappings or {}
    semantic_context = semantic_context or {}
    empty = EventMappingSupport()
    head = mappings.get("head", empty)
    tail = mappings.get("tail", empty)
    reverse_head = mappings.get("reverse_head", empty)
    reverse_tail = mappings.get("reverse_tail", empty)
    direct_head = mention_similarity(str(row["head_text_a"]), str(row["head_text_b"]))
    direct_tail = mention_similarity(str(row["tail_text_a"]), str(row["tail_text_b"]))
    reversed_head = mention_similarity(str(row["head_text_a"]), str(row["tail_text_b"]))
    reversed_tail = mention_similarity(str(row["tail_text_a"]), str(row["head_text_b"]))
    reverse_text = min(reversed_head, reversed_tail)
    direct_text = min(direct_head, direct_tail)
    reverse_mapping = min(reverse_head.support, reverse_tail.support)
    direct_mapping = min(head.support, tail.support)
    reverse_conflict = (
        max(reverse_text, reverse_mapping) > max(direct_text, direct_mapping) + 0.12
        and max(reverse_text, reverse_mapping) >= 0.42
    )
    endpoint_pairs = (
        (str(row["head_text_a"]), str(row["head_text_b"])),
        (str(row["tail_text_a"]), str(row["tail_text_b"])),
    )
    return RelationFeatureRecord(
        direct_head_text=direct_head,
        direct_tail_text=direct_tail,
        reverse_head_text=reversed_head,
        reverse_tail_text=reversed_tail,
        evidence_similarity=float(
            character_dice_similarity(
                str(row.get("evidence_sentence_a", "")), str(row.get("evidence_sentence_b", ""))
            )
        ),
        relation_type_match=float(
            str(row["relation_type_a"]).upper() == str(row["relation_type_b"]).upper()
        ),
        relation_role_compatible=float(
            relation_role_compatible(str(row["relation_type_a"]), str(row["relation_type_b"]))
        ),
        modality_match=float(row.get("modality_a") == row.get("modality_b")),
        head_mapping_support=float(head.support),
        tail_mapping_support=float(tail.support),
        reverse_head_mapping_support=float(reverse_head.support),
        reverse_tail_mapping_support=float(reverse_tail.support),
        head_forbidden_risk=float(head.forbidden_risk),
        tail_forbidden_risk=float(tail.forbidden_risk),
        numeral_conflict=float(
            any(numeral_conflict(left, right) for left, right in endpoint_pairs)
        ),
        negation_conflict=float(
            any(negation_conflict(left, right) for left, right in endpoint_pairs)
        ),
        maintenance_state_conflict=float(
            any(maintenance_state_conflict(left, right) for left, right in endpoint_pairs)
        ),
        reverse_direction_conflict=float(reverse_conflict),
        bge_direct_head_text=float(semantic_context.get("bge_direct_head_text", 0.0)),
        bge_direct_tail_text=float(semantic_context.get("bge_direct_tail_text", 0.0)),
        bge_reverse_head_text=float(semantic_context.get("bge_reverse_head_text", 0.0)),
        bge_reverse_tail_text=float(semantic_context.get("bge_reverse_tail_text", 0.0)),
        bge_evidence_similarity=float(semantic_context.get("bge_evidence_similarity", 0.0)),
        relation_semantic_similarity=float(
            semantic_context.get("relation_semantic_similarity", 0.0)
        ),
        action_semantic_similarity=float(semantic_context.get("action_semantic_similarity", 0.0)),
        action_semantic_available=float(semantic_context.get("action_semantic_available", 0.0)),
        narrative_position_similarity=float(
            semantic_context.get("narrative_position_similarity", 0.0)
        ),
    )


def final_equivalence_score(features: RelationFeatureRecord) -> float:
    """Directed non-compensatory score used by the final Section 3.5 method.

    Direct BGE support, lexical endpoint support, and relation/action semantics
    are all required. A strong endpoint or action expression therefore cannot
    compensate for a weak second endpoint.
    """
    semantic_support = features.relation_semantic_similarity
    if features.action_semantic_available:
        semantic_support = max(semantic_support, features.action_semantic_similarity)
    lexical_floor = 0.45 + 0.55 * features.direct_endpoint_support
    semantic_floor = 0.35 + 0.65 * semantic_support
    return float(
        features.relation_type_match
        * min(
            features.bge_direct_endpoint_support,
            lexical_floor,
            semantic_floor,
        )
    )


def final_conflict_score(features: RelationFeatureRecord) -> float:
    """Risk-prioritized directional/state/numerical/polarity evidence score."""
    direct = features.bge_direct_endpoint_support
    reverse = features.bge_reverse_endpoint_support
    shared_anchor = max(features.direct_head_text, features.direct_tail_text)
    # Numerical and polarity cues become relation-level conflicts only when the
    # two concrete endpoint expressions are already lexically aligned. BGE
    # proximity alone is intentionally insufficient for these hard cues.
    aligned_support = features.direct_endpoint_support
    reverse_score = features.relation_type_match * (0.70 * (reverse - direct) + 0.30 * reverse) - (
        1.0 - features.relation_type_match
    )
    state_score = features.maintenance_state_conflict * (0.30 + 0.25 * shared_anchor)
    numeral_score = (
        features.numeral_conflict * (0.30 + 0.25 * shared_anchor)
        if aligned_support >= 0.55
        else 0.0
    )
    polarity_score = (
        features.negation_conflict * (0.30 + 0.25 * shared_anchor)
        if aligned_support >= 0.45
        else 0.0
    )
    return float(max(reverse_score, state_score, numeral_score, polarity_score))


def conflict_reasons(features: RelationFeatureRecord, *, full: bool = True) -> tuple[str, ...]:
    """Return semantic contradictions, not every reason that forbids merging.

    An endpoint-level ``forbidden_merge`` prediction means that two endpoint
    mentions must remain distinct.  It is an equivalence veto, but by itself it
    does not prove that the two relation facts contradict one another.
    """
    reasons: list[str] = []
    shared_anchor = max(
        features.direct_head_text,
        features.direct_tail_text,
        features.head_mapping_support,
        features.tail_mapping_support,
    )
    if features.reverse_direction_conflict:
        reasons.append("reverse_direction")
    if features.maintenance_state_conflict and shared_anchor >= 0.45:
        reasons.append("maintenance_state")
    aligned_support = max(features.direct_endpoint_support, features.mapping_endpoint_support)
    if full and features.numeral_conflict and aligned_support >= 0.55:
        reasons.append("numeral_conflict")
    if full and features.negation_conflict and aligned_support >= 0.45:
        reasons.append("negation_conflict")
    return tuple(reasons)


def merge_veto_reasons(features: RelationFeatureRecord, *, full: bool = True) -> tuple[str, ...]:
    """Return all hard reasons that make an equivalence merge inadmissible."""
    reasons = list(conflict_reasons(features, full=full))
    aligned_support = max(features.direct_endpoint_support, features.mapping_endpoint_support)
    if (
        full
        and max(features.head_forbidden_risk, features.tail_forbidden_risk) >= 0.75
        and features.direct_endpoint_support >= 0.65
    ):
        reasons.append("endpoint_forbidden")
    if full and features.numeral_conflict and aligned_support >= 0.55:
        reasons.append("numeral_mismatch")
    if full and features.negation_conflict and aligned_support >= 0.45:
        reasons.append("polarity_mismatch")
    return tuple(dict.fromkeys(reasons))


def complementary_vector(features: RelationFeatureRecord, *, mapping_aware: bool) -> list[float]:
    vector = [
        features.bge_evidence_similarity,
        features.relation_semantic_similarity,
        features.action_semantic_similarity,
        features.action_semantic_available,
        max(features.bge_direct_head_text, features.bge_direct_tail_text),
        min(features.bge_direct_head_text, features.bge_direct_tail_text),
        max(features.bge_reverse_head_text, features.bge_reverse_tail_text),
        min(features.bge_reverse_head_text, features.bge_reverse_tail_text),
        features.narrative_position_similarity,
        features.evidence_similarity,
        max(features.direct_head_text, features.direct_tail_text),
        min(features.direct_head_text, features.direct_tail_text),
        features.relation_type_match,
        features.relation_role_compatible,
        max(features.reverse_head_text, features.reverse_tail_text),
        features.modality_match,
    ]
    if mapping_aware:
        vector.extend(
            [
                features.shared_endpoint_support,
                features.mapping_endpoint_support,
                max(features.head_mapping_support, features.tail_mapping_support),
                max(features.reverse_head_mapping_support, features.reverse_tail_mapping_support),
                max(features.head_forbidden_risk, features.tail_forbidden_risk),
            ]
        )
    return vector


class ComplementaryRelationClassifier:
    """Classify residual relation pairs as complementary or unrelated.

    Conflict and equivalence gates must be applied before this classifier. The
    threshold is selected from case-group out-of-fold predictions when groups
    are supplied.
    """

    def __init__(self, *, seed: int = 42, mapping_aware: bool = True) -> None:
        self.seed = int(seed)
        self.mapping_aware = bool(mapping_aware)
        self.threshold = 0.5
        self.model: Any | None = None

    def _new_model(self) -> Any:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                C=0.5,
                max_iter=4000,
                random_state=self.seed,
            ),
        )

    def _matrix(self, features: Sequence[RelationFeatureRecord]) -> np.ndarray:
        return np.asarray(
            [
                complementary_vector(feature, mapping_aware=self.mapping_aware)
                for feature in features
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
        feasible: list[tuple[tuple[float, ...], float]] = []
        fallback: list[tuple[tuple[float, ...], float]] = []
        for threshold in np.linspace(0.10, 0.90, 81):
            prediction = probabilities >= threshold
            macro_f1 = float(f1_score(labels, prediction, average="macro", zero_division=0))
            false_association = float(np.mean(prediction[~labels])) if np.any(~labels) else 0.0
            fallback.append(((macro_f1 - 0.25 * false_association, macro_f1), float(threshold)))
            if false_association <= 0.40:
                feasible.append(
                    ((macro_f1, -false_association, -abs(threshold - 0.5)), float(threshold))
                )
        return max(feasible or fallback, key=lambda item: item[0])[1]

    def fit(
        self,
        features: Sequence[RelationFeatureRecord],
        labels: Sequence[str],
        *,
        groups: Sequence[str] | None = None,
    ) -> ComplementaryRelationClassifier:
        mask = np.asarray([label in {"complementary", "unrelated"} for label in labels])
        if not np.any(mask):
            raise ValueError("training data must include complementary and unrelated rows")
        matrix = self._matrix(features)[mask]
        target = np.asarray(labels, dtype=object)[mask] == "complementary"
        if len(np.unique(target)) != 2:
            raise ValueError("both complementary and unrelated labels are required")
        if groups is not None:
            group_array = np.asarray(groups)[mask]
            class_group_counts = [
                len(np.unique(group_array[target == value])) for value in (False, True)
            ]
            folds = min(4, *class_group_counts)
            if folds >= 2:
                oof = np.zeros(len(target), dtype=np.float64)
                splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=self.seed + 1000)
                for train, valid in splitter.split(matrix, target, group_array):
                    model = self._new_model()
                    model.fit(matrix[train], target[train])
                    oof[valid] = model.predict_proba(matrix[valid])[:, 1]
                self.threshold = self._select_threshold(target, oof)
        self.model = self._new_model()
        self.model.fit(matrix, target)
        return self

    def predict_probability(self, features: Sequence[RelationFeatureRecord]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit must be called before predict_probability")
        return self.model.predict_proba(self._matrix(features))[:, 1]


def decide_relation_state(
    row: Mapping[str, Any],
    features: RelationFeatureRecord,
    mappings: Mapping[str, EventMappingSupport],
    *,
    conflict_threshold: float,
    equivalent_threshold: float,
    complementary_probability: float,
    complementary_threshold: float,
    full_conflict_gate: bool = True,
    abstention_margin: float | None = None,
) -> RelationStateDecision:
    reasons = conflict_reasons(features, full=full_conflict_gate)
    merge_veto = merge_veto_reasons(features, full=full_conflict_gate)
    conflict_score = final_conflict_score(features)
    score = final_equivalence_score(features)
    if conflict_score >= conflict_threshold:
        state = "conflicting"
        reasons = reasons or ("conflict_score_threshold",)
        confidence = min(1.0, 0.5 + abs(conflict_score - conflict_threshold))
    elif score >= equivalent_threshold and not merge_veto:
        state = "equivalent"
        reasons = ("both_endpoints_admissible", "relation_type_match")
        confidence = min(1.0, 0.5 + abs(score - equivalent_threshold))
    elif (
        abstention_margin is not None
        and abs(complementary_probability - complementary_threshold) <= abstention_margin
    ):
        state = "uncertain"
        reasons = ("complementary_unrelated_margin",)
        confidence = 1.0 - abs(complementary_probability - complementary_threshold)
    elif complementary_probability >= complementary_threshold:
        state = "complementary"
        reasons = tuple(merge_veto) + ("shared_case_evidence", "distinct_fact_scope")
        confidence = complementary_probability
    else:
        state = "unrelated"
        reasons = ("insufficient_endpoint_correspondence",)
        confidence = 1.0 - complementary_probability
    provenance = (
        {
            "document_id": row.get("document_id_a"),
            "relation_id": row.get("relation_id_a"),
            "evidence_sentence": row.get("evidence_sentence_a"),
        },
        {
            "document_id": row.get("document_id_b"),
            "relation_id": row.get("relation_id_b"),
            "evidence_sentence": row.get("evidence_sentence_b"),
        },
    )
    return RelationStateDecision(
        relation_a_id=str(row.get("relation_id_a", "")),
        relation_b_id=str(row.get("relation_id_b", "")),
        predicted_state=state,
        graph_action=STATE_TO_ACTION[state],
        equivalence_score=score,
        complementary_probability=float(complementary_probability),
        confidence=float(confidence),
        reason_codes=tuple(reasons),
        endpoint_mapping_support={key: value.to_dict() for key, value in mappings.items()},
        provenance=provenance,
    )
