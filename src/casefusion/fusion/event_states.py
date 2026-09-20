"""Evidence-state-guided event fusion under case constraints."""

from __future__ import annotations

import collections
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from casefusion.text import character_dice_similarity, domain_tokens


def normalize_mention(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[\s，。；：、（）()\-_/]", "", value)
    return re.sub(r"^(?:故障|该|此|现有|相关|重新|局部|原)", "", value)


def mention_similarity(left: str, right: str) -> float:
    """Domain-oriented mention similarity used only to bind relation endpoints."""
    left_norm = normalize_mention(left)
    right_norm = normalize_mention(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    dice = float(character_dice_similarity(left_norm, right_norm))
    if left_norm in right_norm or right_norm in left_norm:
        containment = min(len(left_norm), len(right_norm)) / max(len(left_norm), len(right_norm))
        return max(dice, 0.82 + 0.18 * containment)
    return dice


@dataclass(frozen=True)
class EventMappingSupport:
    """Evidence that two relation endpoints denote the same canonical event."""

    support: float = 0.0
    forbidden_risk: float = 0.0
    mention_fit: float = 0.0
    predicted_state: str = "unmapped"
    source_annotation_id: str | None = None
    source: str = "missing"

    @property
    def available(self) -> bool:
        return self.source_annotation_id is not None or self.source != "missing"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fallback_mapping_support(
    endpoint_a: str,
    endpoint_b: str,
    *,
    type_compatible: bool = True,
) -> EventMappingSupport:
    """Create conservative endpoint support when no persisted mapping exists."""
    mention_fit = mention_similarity(endpoint_a, endpoint_b)
    support = mention_fit if type_compatible else 0.0
    return EventMappingSupport(
        support=float(support),
        forbidden_risk=float(0.0 if type_compatible else mention_fit),
        mention_fit=float(mention_fit),
        predicted_state="equivalent_entity" if support >= 0.72 else "related_not_equivalent",
        source="frozen_event_mapping_fallback",
    )


CANONICAL_EVENT_STATES = ("equivalent", "related", "forbidden")
_STATE_ALIASES = {
    "equivalent_entity": "equivalent",
    "related_not_equivalent": "related",
    "forbidden_merge": "forbidden",
}
_STATE_MARKERS = (
    "分解",
    "拆下",
    "取出",
    "装配",
    "复装",
    "更换",
    "修复",
    "返工",
    "检查",
    "复查",
    "试验",
    "验证",
    "紧固",
    "调整",
    "清理",
    "润滑",
    "盘转",
    "损伤",
    "裂纹",
    "泄漏",
    "松动",
    "磨损",
    "断裂",
    "变形",
    "异响",
    "合格",
)
_SCOPE_MARKERS = ("及", "和", "并", "同时", "全部", "逐", "整套", "局部", "分别")


@dataclass(frozen=True, slots=True)
class EventFusionConfig:
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    inner_folds: int = 4
    trees_per_model: int = 300
    equivalent_threshold_grid: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75)
    forbidden_threshold_grid: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    minimum_equivalent_precision: float = 0.90
    maximum_forbidden_violation_rate: float = 0.10


@dataclass(frozen=True, slots=True)
class EventStateDecision:
    predicted_state: str
    graph_action: str
    equivalent_probability: float
    forbidden_probability: float
    hard_conflict: bool


def _field(row: Mapping[str, Any], canonical: str, alternate: str) -> str:
    if canonical in row:
        return str(row[canonical])
    if alternate in row:
        return str(row[alternate])
    raise KeyError(f"event pair is missing {canonical!r}")


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in domain_tokens(text)
        if len(token) > 1 or any(character.isdigit() for character in token)
    }


def _fit_idf(texts: Sequence[str]) -> tuple[dict[str, float], float]:
    frequencies: collections.Counter[str] = collections.Counter()
    for text in texts:
        frequencies.update(_token_set(text))
    total = max(len(texts), 1)
    return (
        {
            token: math.log((total + 1) / (frequency + 1)) + 1.0
            for token, frequency in frequencies.items()
        },
        math.log(total + 1) + 1.0,
    )


def _directed_coverage(
    source: str, target: str, idf: Mapping[str, float], unknown_idf: float
) -> float:
    source_tokens = _token_set(source)
    target_tokens = _token_set(target)
    denominator = sum(idf.get(token, unknown_idf) for token in source_tokens)
    if not denominator:
        return 0.0
    numerator = sum(idf.get(token, unknown_idf) for token in source_tokens & target_tokens)
    return float(numerator / denominator)


def _numeral_conflict(left: str, right: str) -> bool:
    pattern = r"\d+(?:\.\d+)?"
    left_values = set(re.findall(pattern, left))
    right_values = set(re.findall(pattern, right))
    return bool(left_values and right_values and left_values != right_values)


def _negation_conflict(left: str, right: str) -> bool:
    inspection = r"(?:检查|复查|确认|验证)?(?:有无|是否)"
    left = re.sub(inspection, "", left)
    right = re.sub(inspection, "", right)
    negative = re.compile(r"(?:无|未|没有|不得|不应|不能|消失|为零)")
    return bool(negative.search(left)) != bool(negative.search(right))


def _maintenance_state(text: str) -> str:
    if re.search(r"(?:有无|是否|检查|复查|确认|验证|试验)", text):
        return "inspection"
    if re.search(r"(?:无法|不能|未|没有|为零|无压力|不合格|失效)", text):
        return "negative"
    if re.search(r"(?:采用新|更换|修复|完成装配|合格|恢复正常|有效防松)", text):
        return "repaired"
    return "neutral"


def event_hard_conflict(row: Mapping[str, Any]) -> bool:
    left = _field(row, "event_text_a", "entity_text_a")
    right = _field(row, "event_text_b", "entity_text_b")
    left_type = _field(row, "event_type_a", "entity_type_a")
    right_type = _field(row, "event_type_b", "entity_type_b")
    states = {_maintenance_state(left), _maintenance_state(right)}
    return bool(
        left_type != right_type
        or _numeral_conflict(left, right)
        or _negation_conflict(left, right)
        or states == {"negative", "repaired"}
    )


class EventFeatureEncoder:
    """Fit and transform the semantic, role, state, and scope evidence vector."""

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> EventFeatureEncoder:
        event_texts = [
            _field(row, f"event_text_{side}", f"entity_text_{side}")
            for row in rows
            for side in ("a", "b")
        ]
        evidence_texts = [
            _field(row, f"evidence_text_{side}", f"evidence_sentence_{side}")
            for row in rows
            for side in ("a", "b")
        ]
        self.event_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), sublinear_tf=True
        ).fit(event_texts)
        self.evidence_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), sublinear_tf=True
        ).fit(evidence_texts)
        self.idf, self.unknown_idf = _fit_idf(event_texts + evidence_texts)
        return self

    @staticmethod
    def _paired_cosine(
        vectorizer: TfidfVectorizer, left: Sequence[str], right: Sequence[str]
    ) -> np.ndarray:
        left_matrix = vectorizer.transform(left)
        right_matrix = vectorizer.transform(right)
        return np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        left_events = [_field(row, "event_text_a", "entity_text_a") for row in rows]
        right_events = [_field(row, "event_text_b", "entity_text_b") for row in rows]
        left_evidence = [_field(row, "evidence_text_a", "evidence_sentence_a") for row in rows]
        right_evidence = [_field(row, "evidence_text_b", "evidence_sentence_b") for row in rows]
        event_cosine = self._paired_cosine(self.event_vectorizer, left_events, right_events)
        evidence_cosine = self._paired_cosine(
            self.evidence_vectorizer, left_evidence, right_evidence
        )
        features: list[list[float]] = []
        gates: list[bool] = []
        for index, row in enumerate(rows):
            left = left_events[index]
            right = right_events[index]
            left_context = left_evidence[index]
            right_context = right_evidence[index]
            left_to_right = _directed_coverage(left, right, self.idf, self.unknown_idf)
            right_to_left = _directed_coverage(right, left, self.idf, self.unknown_idf)
            context_left_to_right = _directed_coverage(
                left_context, right_context, self.idf, self.unknown_idf
            )
            context_right_to_left = _directed_coverage(
                right_context, left_context, self.idf, self.unknown_idf
            )
            left_markers = {marker for marker in _STATE_MARKERS if marker in left}
            right_markers = {marker for marker in _STATE_MARKERS if marker in right}
            marker_union = left_markers | right_markers
            features.append(
                [
                    character_dice_similarity(left, right),
                    float(event_cosine[index]),
                    float(evidence_cosine[index]),
                    float(
                        _field(row, "event_type_a", "entity_type_a")
                        == _field(row, "event_type_b", "entity_type_b")
                    ),
                    float(str(row.get("modality_a", "")) == str(row.get("modality_b", ""))),
                    float(not _numeral_conflict(left, right)),
                    float(_negation_conflict(left, right)),
                    min(len(left), len(right)) / max(len(left), len(right), 1),
                    left_to_right,
                    right_to_left,
                    min(left_to_right, right_to_left),
                    max(left_to_right, right_to_left),
                    abs(left_to_right - right_to_left),
                    context_left_to_right,
                    context_right_to_left,
                    min(context_left_to_right, context_right_to_left),
                    abs(context_left_to_right - context_right_to_left),
                    len(left_markers & right_markers) / len(marker_union) if marker_union else 1.0,
                    abs(
                        sum(marker in left for marker in _SCOPE_MARKERS)
                        - sum(marker in right for marker in _SCOPE_MARKERS)
                    ),
                    float(left in right or right in left),
                    float(normalize_mention(left) == normalize_mention(right)),
                    abs(len(left) - len(right)) / 50.0,
                ]
            )
            gates.append(event_hard_conflict(row))
        return np.asarray(features, dtype=np.float64), np.asarray(gates, dtype=bool)


def apply_event_state_rule(
    equivalent_probability: np.ndarray,
    forbidden_probability: np.ndarray,
    hard_conflicts: np.ndarray,
    *,
    equivalent_threshold: float,
    forbidden_threshold: float,
) -> np.ndarray:
    equivalent = (equivalent_probability >= equivalent_threshold) & ~hard_conflicts
    return np.where(
        equivalent,
        "equivalent",
        np.where(forbidden_probability >= forbidden_threshold, "forbidden", "related"),
    ).astype(object)


def _state_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    precision, recall, scores, _ = precision_recall_fscore_support(
        labels, predictions, labels=CANONICAL_EVENT_STATES, zero_division=0
    )
    by_state = {
        state: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(scores[index]),
        }
        for index, state in enumerate(CANONICAL_EVENT_STATES)
    }
    forbidden_mask = labels == "forbidden"
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "equivalent_precision": by_state["equivalent"]["precision"],
        "equivalent_f1": by_state["equivalent"]["f1"],
        "related_f1": by_state["related"]["f1"],
        "forbidden_f1": by_state["forbidden"]["f1"],
        "forbidden_violation_rate": float(np.mean(predictions[forbidden_mask] == "equivalent"))
        if np.any(forbidden_mask)
        else 0.0,
    }


def select_event_thresholds(
    labels: Sequence[str],
    equivalent_probability: np.ndarray,
    forbidden_probability: np.ndarray,
    hard_conflicts: np.ndarray,
    *,
    config: EventFusionConfig | None = None,
) -> tuple[float, float]:
    active = config or EventFusionConfig()
    canonical = np.asarray([_STATE_ALIASES.get(label, label) for label in labels], dtype=object)
    candidates: list[tuple[tuple[float, ...], float, float]] = []
    feasible: list[tuple[tuple[float, ...], float, float]] = []
    for equivalent_threshold in active.equivalent_threshold_grid:
        for forbidden_threshold in active.forbidden_threshold_grid:
            predictions = apply_event_state_rule(
                equivalent_probability,
                forbidden_probability,
                hard_conflicts,
                equivalent_threshold=equivalent_threshold,
                forbidden_threshold=forbidden_threshold,
            )
            metrics = _state_metrics(canonical, predictions)
            key = (
                metrics["macro_f1"],
                min(metrics["related_f1"], metrics["forbidden_f1"]),
                metrics["equivalent_f1"],
                -metrics["forbidden_violation_rate"],
            )
            item = (key, equivalent_threshold, forbidden_threshold)
            candidates.append(item)
            if (
                metrics["equivalent_precision"] >= active.minimum_equivalent_precision
                and metrics["forbidden_violation_rate"] <= active.maximum_forbidden_violation_rate
            ):
                feasible.append(item)
    _, equivalent_threshold, forbidden_threshold = max(
        feasible or candidates, key=lambda item: item[0]
    )
    return float(equivalent_threshold), float(forbidden_threshold)


class EventStateFusionModel:
    """Two-channel ExtraTrees ensemble with case-group threshold selection."""

    def __init__(self, config: EventFusionConfig | None = None) -> None:
        self.config = config or EventFusionConfig()
        self.encoder = EventFeatureEncoder()
        self.equivalent_models: list[ExtraTreesClassifier] = []
        self.forbidden_models: list[ExtraTreesClassifier] = []
        self.equivalent_threshold = 0.65
        self.forbidden_threshold = 0.45

    def _model(self, seed: int) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=self.config.trees_per_model,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    def fit(
        self,
        rows: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        *,
        groups: Sequence[str] | None = None,
    ) -> EventStateFusionModel:
        if not rows or len(rows) != len(labels):
            raise ValueError("rows and labels must be non-empty and have equal length")
        canonical_labels = np.asarray(
            [_STATE_ALIASES.get(label, label) for label in labels], dtype=object
        )
        if set(canonical_labels) - set(CANONICAL_EVENT_STATES):
            raise ValueError("event labels must be equivalent, related, or forbidden")
        if set(canonical_labels) != set(CANONICAL_EVENT_STATES):
            raise ValueError("training data must include all three event states")
        if groups is not None:
            if len(groups) != len(rows):
                raise ValueError("groups must have the same length as rows")
            group_array = np.asarray(groups)
            oof_equivalent = np.zeros(len(rows), dtype=np.float64)
            oof_forbidden = np.zeros(len(rows), dtype=np.float64)
            oof_gates = np.zeros(len(rows), dtype=bool)
            folds = min(self.config.inner_folds, len(np.unique(group_array)))
            if folds < 2:
                raise ValueError("at least two case groups are required for threshold selection")
            splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=self.config.seeds[0])
            for fold, (train, valid) in enumerate(
                splitter.split(np.zeros(len(rows)), canonical_labels == "equivalent", group_array)
            ):
                train_rows = [rows[index] for index in train]
                valid_rows = [rows[index] for index in valid]
                encoder = EventFeatureEncoder().fit(train_rows)
                train_matrix, _ = encoder.transform(train_rows)
                valid_matrix, valid_gates = encoder.transform(valid_rows)
                equivalent_model = self._model(self.config.seeds[fold % len(self.config.seeds)])
                equivalent_model.fit(train_matrix, canonical_labels[train] == "equivalent")
                non_equivalent = canonical_labels[train] != "equivalent"
                forbidden_model = self._model(
                    self.config.seeds[fold % len(self.config.seeds)] + 100
                )
                forbidden_model.fit(
                    train_matrix[non_equivalent],
                    canonical_labels[train][non_equivalent] == "forbidden",
                )
                oof_equivalent[valid] = equivalent_model.predict_proba(valid_matrix)[:, 1]
                oof_forbidden[valid] = forbidden_model.predict_proba(valid_matrix)[:, 1]
                oof_gates[valid] = valid_gates
            self.equivalent_threshold, self.forbidden_threshold = select_event_thresholds(
                canonical_labels, oof_equivalent, oof_forbidden, oof_gates, config=self.config
            )

        self.encoder.fit(rows)
        matrix, _ = self.encoder.transform(rows)
        non_equivalent = canonical_labels != "equivalent"
        self.equivalent_models = []
        self.forbidden_models = []
        for seed in self.config.seeds:
            equivalent_model = self._model(seed)
            equivalent_model.fit(matrix, canonical_labels == "equivalent")
            forbidden_model = self._model(seed + 100)
            forbidden_model.fit(
                matrix[non_equivalent], canonical_labels[non_equivalent] == "forbidden"
            )
            self.equivalent_models.append(equivalent_model)
            self.forbidden_models.append(forbidden_model)
        return self

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[EventStateDecision]:
        if not self.equivalent_models or not self.forbidden_models:
            raise RuntimeError("fit must be called before predict")
        matrix, gates = self.encoder.transform(rows)
        equivalent = np.mean(
            [model.predict_proba(matrix)[:, 1] for model in self.equivalent_models], axis=0
        )
        forbidden = np.mean(
            [model.predict_proba(matrix)[:, 1] for model in self.forbidden_models], axis=0
        )
        states = apply_event_state_rule(
            equivalent,
            forbidden,
            gates,
            equivalent_threshold=self.equivalent_threshold,
            forbidden_threshold=self.forbidden_threshold,
        )
        actions = {"equivalent": "merge", "related": "retain_separate", "forbidden": "block_merge"}
        return [
            EventStateDecision(
                predicted_state=str(state),
                graph_action=actions[str(state)],
                equivalent_probability=float(equivalent[index]),
                forbidden_probability=float(forbidden[index]),
                hard_conflict=bool(gates[index]),
            )
            for index, state in enumerate(states)
        ]
