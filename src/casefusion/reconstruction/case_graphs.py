"""Section 3.6: reconstruct provenance-preserving case graphs.

This module is deliberately model-free.  It consumes frozen case, event and
relation decisions, applies deterministic graph actions, and validates the
result.  In particular, explicit non-equivalence constraints are checked
between whole components before every union so transitive closure cannot
bypass a forbidden decision.
"""

from __future__ import annotations

import collections
import copy
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

EVENT_STATE_ALIASES = {
    "equivalent": "equivalent",
    "equivalent_entity": "equivalent",
    "related": "related",
    "related_not_equivalent": "related",
    "forbidden": "forbidden",
    "forbidden_merge": "forbidden",
    "uncertain": "uncertain",
    "review": "uncertain",
}
RELATION_STATE_ALIASES = {
    "equivalent": "equivalent",
    "equivalent_relation": "equivalent",
    "complementary": "complementary",
    "complementary_fact": "complementary",
    "conflicting": "conflicting",
    "conflicting_fact": "conflicting",
    "unrelated": "unrelated",
    "unrelated_fact": "unrelated",
    "uncertain": "uncertain",
    "review": "uncertain",
}
CONFLICT_REASON_CODES = {
    "numeral_conflict",
    "numeric_conflict",
    "negation_conflict",
    "maintenance_state",
    "maintenance_state_conflict",
    "reverse_direction",
    "reverse_direction_conflict",
    "endpoint_forbidden",
}
_SENTENCE_BOUNDARY = re.compile(r"[。！？；\n]")
_DETAIL_TOKEN = re.compile(
    r"(?:\d+(?:\.\d+)?(?:mm|cm|MPa|kPa|N·m|℃|%|号)?|"
    r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9.]+)+|"
    r"未|无|没有|不得|不能|已|重新|更换|修复|报废|合格|异常|损伤|裂纹|泄漏)"
)


class _ConstrainedDisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.members = {value: {value} for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def component(self, value: str) -> set[str]:
        return self.members[self.find(value)]

    def union(self, left: str, right: str) -> str:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return left_root
        keep, remove = sorted((left_root, right_root))
        self.parent[remove] = keep
        self.members[keep].update(self.members.pop(remove))
        return keep

    def components(self) -> list[list[str]]:
        return [sorted(values) for _, values in sorted(self.members.items())]


def _global_id(document_id: str, local_id: Any) -> str:
    value = str(local_id)
    if value.startswith(f"{document_id}_") or value.startswith(f"{document_id}:"):
        return value.replace(":", "_", 1)
    return f"{document_id}_{value}"


def _sentence_evidence(text: str, start: int, end: int) -> str:
    if not text or start < 0 or end <= start or end > len(text):
        return ""
    left = start
    while left > 0 and not _SENTENCE_BOUNDARY.match(text[left - 1]):
        left -= 1
    right = end
    while right < len(text) and not _SENTENCE_BOUNDARY.match(text[right]):
        right += 1
    if right < len(text):
        right += 1
    return text[left:right].strip()


def normalize_local_graphs(
    local_graphs: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Normalize accepted MCPG input variants without rewriting source facts."""
    events: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    documents: dict[str, dict[str, Any]] = {}
    seen_events: set[str] = set()
    seen_relations: set[str] = set()
    for raw_graph in local_graphs:
        graph = dict(raw_graph)
        document_id = str(graph["document_id"])
        if document_id in documents:
            raise ValueError(f"duplicate local graph document_id: {document_id}")
        source_text = str(graph.get("text", ""))
        documents[document_id] = {
            "document_id": document_id,
            "text": source_text,
            "case_cluster_id": graph.get("case_cluster_id"),
        }
        raw_nodes = graph.get("nodes")
        if raw_nodes is None:
            raw_nodes = graph.get("entities", [])
        event_by_local: dict[str, dict[str, Any]] = {}
        for raw_value in raw_nodes:
            raw = dict(raw_value)
            local_id = str(raw.get("event_id", raw.get("entity_id", raw.get("node_id"))))
            source_event_id = _global_id(document_id, local_id)
            if source_event_id in seen_events:
                raise ValueError(f"duplicate source event id: {source_event_id}")
            mention = str(raw.get("text", raw.get("name", "")))
            start, end = int(raw.get("start", -1)), int(raw.get("end", -1))
            event = {
                "source_event_id": source_event_id,
                "local_event_id": local_id,
                "document_id": document_id,
                "event_type": str(raw.get("type", raw.get("entity_type", "UNKNOWN"))).upper(),
                "text": mention,
                "character_span": [start, end],
                "evidence_text": str(
                    raw.get("evidence_sentence") or _sentence_evidence(source_text, start, end)
                ),
            }
            if raw.get("score", raw.get("confidence")) is not None:
                event["extraction_score"] = float(raw.get("score", raw.get("confidence")))
            for key in ("modality", "state", "support_count", "view_consistency_score"):
                if raw.get(key) is not None:
                    event[key] = raw[key]
            events.append(event)
            event_by_local[local_id] = event
            seen_events.add(source_event_id)

        raw_relations = graph.get("relations", [])
        for raw_value in raw_relations:
            raw = dict(raw_value)
            local_id = str(raw.get("relation_id", raw.get("id")))
            source_relation_id = _global_id(document_id, local_id)
            if source_relation_id in seen_relations:
                raise ValueError(f"duplicate source relation id: {source_relation_id}")
            head_local = str(raw.get("head_event_id", raw.get("head", "")))
            tail_local = str(raw.get("tail_event_id", raw.get("tail", "")))
            head_event_id = _global_id(document_id, head_local)
            tail_event_id = _global_id(document_id, tail_local)
            head = event_by_local.get(head_local)
            tail = event_by_local.get(tail_local)
            evidence = str(raw.get("evidence_text", raw.get("evidence_sentence", "")))
            if not evidence and head is not None and tail is not None:
                starts = [head["character_span"][0], tail["character_span"][0]]
                ends = [head["character_span"][1], tail["character_span"][1]]
                evidence = _sentence_evidence(source_text, min(starts), max(ends))
            relation = {
                "source_relation_id": source_relation_id,
                "local_relation_id": local_id,
                "document_id": document_id,
                "source_head_event_id": head_event_id,
                "relation_type": str(
                    raw.get("type", raw.get("relation", raw.get("relation_type", "UNKNOWN")))
                ).upper(),
                "source_tail_event_id": tail_event_id,
                "evidence_text": evidence,
            }
            if raw.get("score", raw.get("confidence")) is not None:
                relation["extraction_score"] = float(raw.get("score", raw.get("confidence")))
            for key in ("modality", "positive_margin", "decode_mode"):
                if raw.get(key) is not None:
                    relation[key] = raw[key]
            relations.append(relation)
            seen_relations.add(source_relation_id)
    return events, relations, documents


def _normalize_relation_decision(
    raw_value: Mapping[str, Any], decision_id: str | None = None
) -> dict[str, Any]:
    raw = copy.deepcopy(dict(raw_value))
    state_value = raw.get(
        "relation_state", raw.get("predicted_state", raw.get("state", "uncertain"))
    )
    state = RELATION_STATE_ALIASES.get(str(state_value), "uncertain")
    reasons = [str(value) for value in raw.get("decision_reasons", raw.get("reason_codes", []))]
    dimensions = [str(value) for value in raw.get("conflict_dimensions", [])]
    if not dimensions:
        dimensions = [value for value in reasons if value in CONFLICT_REASON_CODES]
    return {
        **raw,
        "decision_id": str(decision_id or raw.get("decision_id", raw.get("annotation_id", ""))),
        "relation_a_id": str(raw.get("relation_a_id", raw.get("relation_a", ""))).replace(
            ":", "_", 1
        ),
        "relation_b_id": str(raw.get("relation_b_id", raw.get("relation_b", ""))).replace(
            ":", "_", 1
        ),
        "relation_state": state,
        "confidence": float(raw.get("confidence") or raw.get("equivalence_score") or 0.0),
        "decision_reasons": reasons,
        "conflict_dimensions": dimensions,
    }


def derive_event_decisions_from_relation_decisions(
    source_relations: Iterable[Mapping[str, Any]],
    relation_decisions: Iterable[Mapping[str, Any]],
    *,
    mention_fit_threshold: float = 0.72,
) -> list[dict[str, Any]]:
    """Expose embedded frozen Section-3.4 endpoint states as source-event decisions.

    Section 3.5 persisted actual source relation IDs plus the Section 3.4
    endpoint support it consumed.  This adapter projects those already-made
    decisions back to the concrete local MCPG endpoints.  It does not read a
    supervised labels and does not make a new semantic prediction.
    """
    relation_index = {str(row["source_relation_id"]): dict(row) for row in source_relations}
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    endpoint_specs = (
        ("head", "source_head_event_id", "source_head_event_id"),
        ("tail", "source_tail_event_id", "source_tail_event_id"),
        ("reverse_head", "source_head_event_id", "source_tail_event_id"),
        ("reverse_tail", "source_tail_event_id", "source_head_event_id"),
    )
    for raw_value in relation_decisions:
        decision = _normalize_relation_decision(raw_value)
        left = relation_index.get(decision["relation_a_id"])
        right = relation_index.get(decision["relation_b_id"])
        if left is None or right is None:
            continue
        support_index = decision.get("endpoint_mapping_support", {})
        for endpoint_role, left_key, right_key in endpoint_specs:
            support = dict(support_index.get(endpoint_role, {}) or {})
            mention_fit = float(support.get("mention_fit") or 0.0)
            if mention_fit < mention_fit_threshold:
                continue
            raw_state = str(support.get("predicted_state", "related_not_equivalent"))
            state = EVENT_STATE_ALIASES.get(raw_state, "uncertain")
            event_a = str(left[left_key])
            event_b = str(right[right_key])
            if event_a == event_b:
                continue
            pair = tuple(sorted((event_a, event_b)))
            candidates[pair].append(
                {
                    "event_a_id": pair[0],
                    "event_b_id": pair[1],
                    "event_state": state,
                    "equivalence_support": float(support.get("support") or 0.0),
                    "forbidden_risk": float(support.get("forbidden_risk") or 0.0),
                    "mention_fit": mention_fit,
                    "source_annotation_id": support.get("source_annotation_id"),
                    "source_relation_decision_id": decision.get("decision_id"),
                    "endpoint_role": endpoint_role,
                    "source": support.get("source", "section_3_5_embedded_section_3_4"),
                }
            )

    output: list[dict[str, Any]] = []
    for index, (pair, rows) in enumerate(sorted(candidates.items()), start=1):
        states = {row["event_state"] for row in rows}
        if "forbidden" in states:
            state = "forbidden"
        elif "equivalent" in states:
            state = "equivalent"
        elif "uncertain" in states:
            state = "uncertain"
        else:
            state = "related"
        output.append(
            {
                "decision_id": f"S36_ED{index:06d}",
                "event_a_id": pair[0],
                "event_b_id": pair[1],
                "event_state": state,
                "equivalence_support": max(row["equivalence_support"] for row in rows),
                "forbidden_risk": max(row["forbidden_risk"] for row in rows),
                "hard_veto": state == "forbidden",
                "confidence": max(
                    max(row["equivalence_support"], row["forbidden_risk"], row["mention_fit"])
                    for row in rows
                ),
                "decision_reasons": sorted(
                    {f"embedded_endpoint:{row['endpoint_role']}" for row in rows}
                ),
                "source_decisions": rows,
                "source": "frozen_section_3_4_state_embedded_in_section_3_5",
            }
        )
    return output


def _normalize_event_decision(raw_value: Mapping[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(dict(raw_value))
    raw_state = raw.get("event_state", raw.get("predicted_state", raw.get("state", "uncertain")))
    state = EVENT_STATE_ALIASES.get(str(raw_state), "uncertain")
    return {
        **raw,
        "decision_id": str(raw.get("decision_id", raw.get("annotation_id", ""))),
        "event_a_id": str(raw.get("event_a_id", raw.get("event_a", ""))).replace(":", "_", 1),
        "event_b_id": str(raw.get("event_b_id", raw.get("event_b", ""))).replace(":", "_", 1),
        "event_state": state,
        "equivalence_support": float(raw.get("equivalence_support") or raw.get("score") or 0.0),
        "forbidden_risk": float(raw.get("forbidden_risk") or 0.0),
        "confidence": float(
            raw.get("confidence") or raw.get("equivalence_support") or raw.get("score") or 0.0
        ),
        "hard_veto": bool(raw.get("hard_veto", state == "forbidden")),
        "decision_reasons": [
            str(value) for value in raw.get("decision_reasons", raw.get("reason_codes", []))
        ],
    }


def _rank_source_event(row: Mapping[str, Any]) -> tuple[Any, ...]:
    text = str(row.get("text", ""))
    detail_count = len(_DETAIL_TOKEN.findall(text))
    completeness = len(re.sub(r"\s+", "", text)) + 6 * detail_count
    return (
        -completeness,
        -detail_count,
        -float(row.get("extraction_score") or 0.0),
        str(row["source_event_id"]),
    )


def _rank_source_relation(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -len(str(row.get("evidence_text", ""))),
        -float(row.get("extraction_score") or 0.0),
        str(row["source_relation_id"]),
    )


def _blocking_pairs(
    left_members: Iterable[str],
    right_members: Iterable[str],
    blocked: Mapping[frozenset[str], str],
) -> list[dict[str, Any]]:
    result = []
    for left in left_members:
        for right in right_members:
            state = blocked.get(frozenset((left, right)))
            if state is not None:
                result.append({"pair": sorted((left, right)), "state": state})
    return sorted(result, key=lambda row: row["pair"])


def _build_event_layer(
    case_id: str,
    source_events: Sequence[dict[str, Any]],
    event_decisions: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    event_index = {row["source_event_id"]: row for row in source_events}
    dsu = _ConstrainedDisjointSet(event_index)
    decisions = [
        _normalize_event_decision(row)
        for row in event_decisions
        if str(row.get("event_a_id", row.get("event_a", ""))).replace(":", "_", 1) in event_index
        and str(row.get("event_b_id", row.get("event_b", ""))).replace(":", "_", 1) in event_index
    ]
    blocked = {
        frozenset((row["event_a_id"], row["event_b_id"])): row["event_state"]
        for row in decisions
        if row["event_state"] != "equivalent"
    }
    equivalent = sorted(
        (row for row in decisions if row["event_state"] == "equivalent"),
        key=lambda row: (
            -row["equivalence_support"],
            -row["confidence"],
            row["event_a_id"],
            row["event_b_id"],
        ),
    )
    accepted: list[dict[str, Any]] = []
    blocked_merges: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for decision in equivalent:
        left, right = decision["event_a_id"], decision["event_b_id"]
        if dsu.find(left) == dsu.find(right):
            accepted.append({**decision, "action": "already_connected"})
            continue
        constraints = _blocking_pairs(dsu.component(left), dsu.component(right), blocked)
        if constraints:
            item = {
                "review_type": "blocked_event_equivalence",
                "case_cluster_id": case_id,
                "event_a_id": left,
                "event_b_id": right,
                "blocking_pairs": constraints,
                "decision_trace": decision,
            }
            blocked_merges.append(item)
            review.append(item)
            continue
        dsu.union(left, right)
        accepted.append({**decision, "action": "merge"})

    for decision in decisions:
        if decision["event_state"] == "uncertain":
            review.append(
                {
                    "review_type": "uncertain_event_pair",
                    "case_cluster_id": case_id,
                    "event_a_id": decision["event_a_id"],
                    "event_b_id": decision["event_b_id"],
                    "decision_trace": decision,
                }
            )

    components = sorted(dsu.components(), key=lambda values: values[0])
    source_to_case: dict[str, str] = {}
    case_events: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, member_ids in enumerate(components, start=1):
        case_event_id = f"{case_id}_E{index:04d}"
        rows = [event_index[value] for value in member_ids]
        representative = sorted(rows, key=_rank_source_event)[0]
        for member_id in member_ids:
            source_to_case[member_id] = case_event_id
        merge_trace = [
            row
            for row in accepted
            if row["event_a_id"] in member_ids and row["event_b_id"] in member_ids
        ]
        event = {
            "case_event_id": case_event_id,
            "case_cluster_id": case_id,
            "event_type": representative["event_type"],
            "canonical_text": representative["text"],
            "canonical_source_event_id": representative["source_event_id"],
            "source_event_ids": member_ids,
            "source_document_ids": sorted({row["document_id"] for row in rows}),
            "character_spans": [row["character_span"] for row in rows],
            "evidence_texts": [row["evidence_text"] for row in rows if row.get("evidence_text")],
            "extraction_scores": [
                row["extraction_score"] for row in rows if row.get("extraction_score") is not None
            ],
            "fusion_scores": [row["equivalence_support"] for row in merge_trace],
            "decision_reasons": sorted(
                {reason for row in merge_trace for reason in row["decision_reasons"]}
            ),
            "merge_decisions": merge_trace,
            "review_status": "not_required",
        }
        case_events.append(event)
        provenance.append(
            {
                "case_event_id": case_event_id,
                "case_cluster_id": case_id,
                "sources": [copy.deepcopy(row) for row in rows],
            }
        )
    return case_events, source_to_case, provenance, blocked_merges, review


def _build_relation_layer(
    case_id: str,
    source_relations: Sequence[dict[str, Any]],
    source_to_case_event: Mapping[str, str],
    relation_decisions: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    relation_index = {row["source_relation_id"]: row for row in source_relations}
    mapped = {
        relation_id: {
            **row,
            "head_case_event_id": source_to_case_event.get(row["source_head_event_id"], ""),
            "tail_case_event_id": source_to_case_event.get(row["source_tail_event_id"], ""),
        }
        for relation_id, row in relation_index.items()
    }
    dsu = _ConstrainedDisjointSet(relation_index)
    decisions = [
        _normalize_relation_decision(row)
        for row in relation_decisions
        if str(row.get("relation_a_id", row.get("relation_a", ""))).replace(":", "_", 1)
        in relation_index
        and str(row.get("relation_b_id", row.get("relation_b", ""))).replace(":", "_", 1)
        in relation_index
    ]
    blocked = {
        frozenset((row["relation_a_id"], row["relation_b_id"])): row["relation_state"]
        for row in decisions
        if row["relation_state"] != "equivalent"
    }
    equivalent = sorted(
        (row for row in decisions if row["relation_state"] == "equivalent"),
        key=lambda row: (
            -float(row.get("equivalence_score") or 0.0),
            -row["confidence"],
            row["relation_a_id"],
            row["relation_b_id"],
        ),
    )
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for decision in equivalent:
        left_id, right_id = decision["relation_a_id"], decision["relation_b_id"]
        left, right = mapped[left_id], mapped[right_id]
        compatible = (
            left["head_case_event_id"]
            and left["head_case_event_id"] == right["head_case_event_id"]
            and left["relation_type"] == right["relation_type"]
            and left["tail_case_event_id"] == right["tail_case_event_id"]
        )
        if not compatible:
            item = {
                "review_type": "equivalent_relation_endpoint_mismatch",
                "case_cluster_id": case_id,
                "relation_a_id": left_id,
                "relation_b_id": right_id,
                "mapped_a": [
                    left["head_case_event_id"],
                    left["relation_type"],
                    left["tail_case_event_id"],
                ],
                "mapped_b": [
                    right["head_case_event_id"],
                    right["relation_type"],
                    right["tail_case_event_id"],
                ],
                "decision_trace": decision,
            }
            suppressed.append(item)
            review.append(item)
            continue
        if dsu.find(left_id) == dsu.find(right_id):
            accepted.append({**decision, "action": "already_connected"})
            continue
        constraints = _blocking_pairs(dsu.component(left_id), dsu.component(right_id), blocked)
        if constraints:
            item = {
                "review_type": "blocked_relation_equivalence",
                "case_cluster_id": case_id,
                "relation_a_id": left_id,
                "relation_b_id": right_id,
                "blocking_pairs": constraints,
                "decision_trace": decision,
            }
            suppressed.append(item)
            review.append(item)
            continue
        dsu.union(left_id, right_id)
        accepted.append({**decision, "action": "merge"})

    components = sorted(dsu.components(), key=lambda values: values[0])
    source_to_case_relation: dict[str, str] = {}
    case_relations: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, member_ids in enumerate(components, start=1):
        case_relation_id = f"{case_id}_R{index:04d}"
        rows = [mapped[value] for value in member_ids]
        representative = sorted(rows, key=_rank_source_relation)[0]
        for member_id in member_ids:
            source_to_case_relation[member_id] = case_relation_id
        merge_trace = [
            row
            for row in accepted
            if row["relation_a_id"] in member_ids and row["relation_b_id"] in member_ids
        ]
        relation = {
            "case_relation_id": case_relation_id,
            "case_cluster_id": case_id,
            "head_case_event_id": representative["head_case_event_id"],
            "relation_type": representative["relation_type"],
            "tail_case_event_id": representative["tail_case_event_id"],
            "relation_state": "equivalent" if len(member_ids) > 1 else "retained",
            "source_relation_ids": member_ids,
            "source_document_ids": sorted({row["document_id"] for row in rows}),
            "evidence_texts": [row["evidence_text"] for row in rows if row.get("evidence_text")],
            "modalities": [row["modality"] for row in rows if row.get("modality") is not None],
            "extraction_scores": [
                row["extraction_score"] for row in rows if row.get("extraction_score") is not None
            ],
            "fusion_scores": [
                float(row.get("equivalence_score") or row.get("confidence") or 0.0)
                for row in merge_trace
            ],
            "decision_reasons": sorted(
                {reason for row in merge_trace for reason in row["decision_reasons"]}
            ),
            "merge_decisions": merge_trace,
            "complementary_with": [],
            "conflict_with": [],
            "conflict_dimensions": [],
            "review_status": "not_required",
        }
        case_relations.append(relation)
        provenance.append(
            {
                "case_relation_id": case_relation_id,
                "case_cluster_id": case_id,
                "sources": [copy.deepcopy(relation_index[value]) for value in member_ids],
            }
        )

    final_index = {row["case_relation_id"]: row for row in case_relations}
    pair_trace: list[dict[str, Any]] = []
    for decision in decisions:
        left = source_to_case_relation[decision["relation_a_id"]]
        right = source_to_case_relation[decision["relation_b_id"]]
        state = decision["relation_state"]
        pair_trace.append(
            {
                "left_case_relation_id": left,
                "right_case_relation_id": right,
                "relation_state": state,
                "source_relation_ids": [decision["relation_a_id"], decision["relation_b_id"]],
                "decision_id": decision["decision_id"],
                "confidence": decision["confidence"],
                "decision_reasons": decision["decision_reasons"],
                "conflict_dimensions": decision["conflict_dimensions"],
            }
        )
        if state == "complementary" and left != right:
            final_index[left]["complementary_with"].append(right)
            final_index[right]["complementary_with"].append(left)
        elif state == "conflicting" and left != right:
            final_index[left]["conflict_with"].append(right)
            final_index[right]["conflict_with"].append(left)
            for relation_id in (left, right):
                final_index[relation_id]["conflict_dimensions"].extend(
                    decision["conflict_dimensions"]
                )
        elif state == "uncertain":
            for relation_id in {left, right}:
                final_index[relation_id]["review_status"] = "review_required"
            review.append(
                {
                    "review_type": "uncertain_relation_pair",
                    "case_cluster_id": case_id,
                    "relation_a_id": decision["relation_a_id"],
                    "relation_b_id": decision["relation_b_id"],
                    "decision_trace": decision,
                }
            )
    for relation in case_relations:
        relation["complementary_with"] = sorted(set(relation["complementary_with"]))
        relation["conflict_with"] = sorted(set(relation["conflict_with"]))
        relation["conflict_dimensions"] = sorted(set(relation["conflict_dimensions"]))
    return case_relations, provenance, pair_trace, review, source_to_case_relation


def validate_case_graph(
    graph: Mapping[str, Any],
    source_events: Sequence[Mapping[str, Any]],
    source_relations: Sequence[Mapping[str, Any]],
    event_decisions: Sequence[Mapping[str, Any]],
    relation_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the sixteen hard Section-3.6 reconstruction invariants."""
    case_id = str(graph["case_cluster_id"])
    document_ids = set(map(str, graph["source_document_ids"]))
    events = {str(row["case_event_id"]): row for row in graph["events"]}
    relations = {str(row["case_relation_id"]): row for row in graph["relations"]}
    event_map = {str(key): str(value) for key, value in graph["event_mapping"].items()}
    source_event_index = {str(row["source_event_id"]): row for row in source_events}
    source_relation_index = {str(row["source_relation_id"]): row for row in source_relations}
    relation_map = {
        source_id: relation_id
        for relation_id, row in relations.items()
        for source_id in row["source_relation_ids"]
    }
    normalized_event_decisions = [_normalize_event_decision(row) for row in event_decisions]
    normalized_relation_decisions = [
        _normalize_relation_decision(row) for row in relation_decisions
    ]

    dangling_heads = sorted(
        row["case_relation_id"]
        for row in relations.values()
        if row["head_case_event_id"] not in events
    )
    dangling_tails = sorted(
        row["case_relation_id"]
        for row in relations.values()
        if row["tail_case_event_id"] not in events
    )
    empty_event_sources = sorted(
        row["case_event_id"] for row in events.values() if not row["source_event_ids"]
    )
    empty_relation_sources = sorted(
        row["case_relation_id"] for row in relations.values() if not row["source_relation_ids"]
    )
    cross_event = sorted(
        row["case_event_id"]
        for row in events.values()
        if not set(row["source_document_ids"]).issubset(document_ids)
    )
    cross_relation = sorted(
        row["case_relation_id"]
        for row in relations.values()
        if not set(row["source_document_ids"]).issubset(document_ids)
    )
    forbidden_violations = sorted(
        [
            [row["event_a_id"], row["event_b_id"]]
            for row in normalized_event_decisions
            if row["event_state"] == "forbidden"
            and event_map.get(row["event_a_id"]) == event_map.get(row["event_b_id"])
            and event_map.get(row["event_a_id"]) is not None
        ]
    )
    conflicting_collapses = sorted(
        [
            [row["relation_a_id"], row["relation_b_id"]]
            for row in normalized_relation_decisions
            if row["relation_state"] == "conflicting"
            and relation_map.get(row["relation_a_id"]) == relation_map.get(row["relation_b_id"])
            and relation_map.get(row["relation_a_id"]) is not None
        ]
    )
    complementary_collapses = sorted(
        [
            [row["relation_a_id"], row["relation_b_id"]]
            for row in normalized_relation_decisions
            if row["relation_state"] == "complementary"
            and relation_map.get(row["relation_a_id"]) == relation_map.get(row["relation_b_id"])
            and relation_map.get(row["relation_a_id"]) is not None
        ]
    )
    incompatible_equivalent_groups: list[str] = []
    for relation_id, row in relations.items():
        facts = [
            source_relation_index[value]
            for value in row["source_relation_ids"]
            if value in source_relation_index
        ]
        signatures = {
            (
                event_map.get(fact["source_head_event_id"]),
                fact["relation_type"],
                event_map.get(fact["source_tail_event_id"]),
            )
            for fact in facts
        }
        if len(signatures) > 1:
            incompatible_equivalent_groups.append(relation_id)
    missing_event_documents = sorted(
        row["source_event_id"]
        for row in source_events
        if not row.get("document_id") or row["document_id"] not in document_ids
    )
    missing_relation_documents = sorted(
        row["source_relation_id"]
        for row in source_relations
        if not row.get("document_id") or row["document_id"] not in document_ids
    )
    missing_event_evidence = sorted(
        row["source_event_id"]
        for row in source_events
        if not row.get("text") and not row.get("evidence_text")
    )
    missing_relation_evidence = sorted(
        row["source_relation_id"] for row in source_relations if not row.get("evidence_text")
    )
    rewritten_sources: list[str] = []
    for event in events.values():
        source_rows = [
            source_event_index[value]
            for value in event["source_event_ids"]
            if value in source_event_index
        ]
        if event["canonical_text"] not in {row["text"] for row in source_rows}:
            rewritten_sources.append(event["case_event_id"])
    missing_event_trace = sorted(
        row["case_event_id"]
        for row in events.values()
        if len(row["source_event_ids"]) > 1 and not row.get("merge_decisions")
    )
    missing_relation_trace = sorted(
        row["case_relation_id"]
        for row in relations.values()
        if len(row["source_relation_ids"]) > 1 and not row.get("merge_decisions")
    )
    unmapped_events = sorted(set(source_event_index) - set(event_map))
    unmapped_relations = sorted(set(source_relation_index) - set(relation_map))

    checks = {
        "01_relation_heads_exist": {"passed": not dangling_heads, "details": dangling_heads},
        "02_relation_tails_exist": {"passed": not dangling_tails, "details": dangling_tails},
        "03_no_dangling_endpoints": {
            "passed": not dangling_heads and not dangling_tails,
            "details": sorted(set(dangling_heads + dangling_tails)),
        },
        "04_events_have_sources": {
            "passed": not empty_event_sources and not unmapped_events,
            "details": {"empty": empty_event_sources, "unmapped": unmapped_events},
        },
        "05_relations_have_sources": {
            "passed": not empty_relation_sources and not unmapped_relations,
            "details": {"empty": empty_relation_sources, "unmapped": unmapped_relations},
        },
        "06_event_case_boundary": {"passed": not cross_event, "details": cross_event},
        "07_relation_case_boundary": {"passed": not cross_relation, "details": cross_relation},
        "08_forbidden_events_not_merged": {
            "passed": not forbidden_violations,
            "details": forbidden_violations,
        },
        "09_conflicting_relations_not_collapsed": {
            "passed": not conflicting_collapses,
            "details": conflicting_collapses,
        },
        "10_complementary_relations_distinct": {
            "passed": not complementary_collapses,
            "details": complementary_collapses,
        },
        "11_equivalent_relation_compatibility": {
            "passed": not incompatible_equivalent_groups,
            "details": incompatible_equivalent_groups,
        },
        "12_event_document_trace": {
            "passed": not missing_event_documents,
            "details": missing_event_documents,
        },
        "13_relation_document_trace": {
            "passed": not missing_relation_documents,
            "details": missing_relation_documents,
        },
        "14_original_evidence_recoverable": {
            "passed": not missing_event_evidence and not missing_relation_evidence,
            "details": {"events": missing_event_evidence, "relations": missing_relation_evidence},
        },
        "15_source_wording_and_values_preserved": {
            "passed": not rewritten_sources,
            "details": rewritten_sources,
        },
        "16_automatic_merges_retain_trace": {
            "passed": not missing_event_trace and not missing_relation_trace,
            "details": {"events": missing_event_trace, "relations": missing_relation_trace},
        },
    }
    failed = [name for name, value in checks.items() if not value["passed"]]
    return {
        "case_cluster_id": case_id,
        "passed": not failed,
        "failed_invariants": failed,
        "checks": checks,
        "counts": {
            "source_events": len(source_events),
            "case_events": len(events),
            "source_relations": len(source_relations),
            "case_relations": len(relations),
        },
    }


def build_case_graphs(
    local_graphs: Iterable[Mapping[str, Any]],
    case_clusters: Mapping[str, str],
    event_fusion_results: Iterable[Mapping[str, Any]],
    relation_fusion_results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Unified model-free reconstruction entry point required by Section 3.6."""
    source_events, source_relations, documents = normalize_local_graphs(local_graphs)
    assignments = {str(document_id): str(case_id) for document_id, case_id in case_clusters.items()}
    missing_assignments = sorted(set(documents) - set(assignments))
    unknown_assignments = sorted(set(assignments) - set(documents))
    if missing_assignments or unknown_assignments:
        raise ValueError(
            f"case/local-graph coverage mismatch: missing={missing_assignments[:10]} "
            f"unknown={unknown_assignments[:10]}"
        )
    events_by_case: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    relations_by_case: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    documents_by_case: dict[str, list[str]] = collections.defaultdict(list)
    event_case: dict[str, str] = {}
    relation_case: dict[str, str] = {}
    for document_id, case_id in assignments.items():
        documents_by_case[case_id].append(document_id)
    for event in source_events:
        case_id = assignments[event["document_id"]]
        events_by_case[case_id].append(event)
        event_case[event["source_event_id"]] = case_id
    for relation in source_relations:
        case_id = assignments[relation["document_id"]]
        relations_by_case[case_id].append(relation)
        relation_case[relation["source_relation_id"]] = case_id

    normalized_events = [_normalize_event_decision(row) for row in event_fusion_results]
    normalized_relations = [_normalize_relation_decision(row) for row in relation_fusion_results]
    event_decisions_by_case: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    relation_decisions_by_case: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    ignored_event_decisions = collections.Counter()
    ignored_relation_decisions = collections.Counter()
    for row in normalized_events:
        left_case, right_case = event_case.get(row["event_a_id"]), event_case.get(row["event_b_id"])
        if left_case is None or right_case is None:
            ignored_event_decisions["unknown_endpoint"] += 1
        elif left_case != right_case:
            ignored_event_decisions["cross_case"] += 1
        else:
            event_decisions_by_case[left_case].append(row)
    for row in normalized_relations:
        left_case = relation_case.get(row["relation_a_id"])
        right_case = relation_case.get(row["relation_b_id"])
        if left_case is None or right_case is None:
            ignored_relation_decisions["unknown_endpoint"] += 1
        elif left_case != right_case:
            ignored_relation_decisions["cross_case"] += 1
        else:
            relation_decisions_by_case[left_case].append(row)

    valid_graphs: list[dict[str, Any]] = []
    invalid_graphs: list[dict[str, Any]] = []
    event_mapping_rows: list[dict[str, Any]] = []
    event_provenance_rows: list[dict[str, Any]] = []
    relation_provenance_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    totals = collections.Counter()
    event_state_counts = collections.Counter(row["event_state"] for row in normalized_events)
    relation_state_counts = collections.Counter(
        row["relation_state"] for row in normalized_relations
    )
    for case_id in sorted(documents_by_case):
        case_events_source = sorted(events_by_case[case_id], key=lambda row: row["source_event_id"])
        case_relations_source = sorted(
            relations_by_case[case_id], key=lambda row: row["source_relation_id"]
        )
        case_event_decisions = event_decisions_by_case[case_id]
        case_relation_decisions = relation_decisions_by_case[case_id]
        case_events, event_mapping, event_provenance, blocked_events, event_review = (
            _build_event_layer(case_id, case_events_source, case_event_decisions)
        )
        case_relations, relation_provenance, relation_trace, relation_review, relation_mapping = (
            _build_relation_layer(
                case_id, case_relations_source, event_mapping, case_relation_decisions
            )
        )
        graph_review = [*event_review, *relation_review]
        for index, item in enumerate(graph_review, start=1):
            item["review_id"] = f"{case_id}_Q{index:04d}"
        graph = {
            "case_cluster_id": case_id,
            "source_document_ids": sorted(documents_by_case[case_id]),
            "events": case_events,
            "relations": case_relations,
            "event_mapping": event_mapping,
            "relation_mapping": relation_mapping,
            "provenance": {
                "events": {row["case_event_id"]: row for row in event_provenance},
                "relations": {row["case_relation_id"]: row for row in relation_provenance},
            },
            "relation_decision_trace": relation_trace,
            "blocked_event_merges": blocked_events,
            "review_queue": graph_review,
            "metadata": {
                "section": "3.6",
                "reconstruction": "confidence_ordered_constrained_union_find",
                "event_merge_order": "equivalence_support_desc_confidence_desc_ids_asc",
                "relation_merge_order": "equivalence_score_desc_confidence_desc_ids_asc",
                "canonical_text_policy": "select_existing_source_mention_only",
                "supervised_identity_required": False,
            },
        }
        validation = validate_case_graph(
            graph,
            case_events_source,
            case_relations_source,
            case_event_decisions,
            case_relation_decisions,
        )
        graph["validation"] = validation
        validation_rows.append(validation)
        if validation["passed"]:
            valid_graphs.append(graph)
        else:
            invalid_graphs.append(graph)
        event_mapping_rows.extend(
            {
                "case_cluster_id": case_id,
                "source_event_id": source_id,
                "case_event_id": target_id,
            }
            for source_id, target_id in sorted(event_mapping.items())
        )
        event_provenance_rows.extend(event_provenance)
        relation_provenance_rows.extend(relation_provenance)
        review_rows.extend(graph_review)
        totals.update(
            {
                "source_events": len(case_events_source),
                "case_events": len(case_events),
                "source_relations": len(case_relations_source),
                "case_relations": len(case_relations),
                "equivalent_event_merges": len(case_events_source) - len(case_events),
                "blocked_event_merges": len(blocked_events),
                "equivalent_relation_merges": len(case_relations_source) - len(case_relations),
                "review_items": len(graph_review),
            }
        )
    summary = {
        "status": "passed" if not invalid_graphs else "failed_validation",
        "scope": "engineering_reconstruction_not_accuracy_evaluation",
        "case_clusters_processed": len(documents_by_case),
        "documents_processed": len(documents),
        **dict(totals),
        "event_decision_states": dict(sorted(event_state_counts.items())),
        "relation_decision_states": dict(sorted(relation_state_counts.items())),
        "complementary_relation_decisions_retained": relation_state_counts["complementary"],
        "conflicting_relation_decisions_retained": relation_state_counts["conflicting"],
        "unrelated_relation_decisions_retained": relation_state_counts["unrelated"],
        "graphs_passing_validation": len(valid_graphs),
        "graphs_failing_validation": len(invalid_graphs),
        "ignored_event_decisions": dict(ignored_event_decisions),
        "ignored_relation_decisions": dict(ignored_relation_decisions),
        "merge_policy": "frozen decisions only; no missing decision is imputed as equivalence",
    }
    validation_report = {
        "passed": not invalid_graphs,
        "graph_count": len(validation_rows),
        "passed_graph_count": len(valid_graphs),
        "failed_graph_count": len(invalid_graphs),
        "failed_case_cluster_ids": [
            row["case_cluster_id"] for row in validation_rows if not row["passed"]
        ],
        "graphs": validation_rows,
    }
    return {
        "case_graphs": valid_graphs,
        "invalid_case_graphs": invalid_graphs,
        "event_mapping": event_mapping_rows,
        "event_provenance": event_provenance_rows,
        "relation_provenance": relation_provenance_rows,
        "review_queue": review_rows,
        "validation_report": validation_report,
        "summary": summary,
    }
