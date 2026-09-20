from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROLES = {"PHENOMENON", "FAILURE", "ROOT_CAUSE", "ACTION", "VERIFICATION"}
RELATIONS = {"CAUSE", "TEMPORAL", "TREAT", "VERIFY"}


@dataclass(frozen=True, slots=True)
class EventNode:
    uid: str
    case_id: str
    document_id: str
    event_id: str
    text: str
    role: str
    start: int
    end: int
    score: float
    source_span_valid: bool
    source_event_ids: tuple[str, ...] = ()
    source_document_ids: tuple[str, ...] = ()
    source_spans: tuple[tuple[int, int], ...] = ()
    canonical_source_event_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RelationEdge:
    uid: str
    case_id: str
    document_id: str
    relation_id: str
    head_uid: str
    tail_uid: str
    relation_type: str
    score: float
    evidence_text: str
    source_relation_ids: tuple[str, ...] = ()
    source_document_ids: tuple[str, ...] = ()
    relation_state: str = "retained"
    complementary_with: tuple[str, ...] = ()
    conflict_with: tuple[str, ...] = ()
    conflict_dimensions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CaseGraph:
    case_id: str
    document_ids: tuple[str, ...]
    nodes: tuple[EventNode, ...]
    edges: tuple[RelationEdge, ...]
    source_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        node_ids = {node.uid for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            errors.append("duplicate_node_uid")
        if len({edge.uid for edge in self.edges}) != len(self.edges):
            errors.append("duplicate_edge_uid")
        for node in self.nodes:
            if node.case_id != self.case_id:
                errors.append(f"cross_case_node:{node.uid}")
            if node.role not in ROLES:
                errors.append(f"unknown_role:{node.uid}:{node.role}")
            source_documents = set(node.source_document_ids or (node.document_id,))
            if not source_documents.issubset(self.document_ids):
                errors.append(f"cross_case_node_source:{node.uid}")
        for edge in self.edges:
            if edge.case_id != self.case_id:
                errors.append(f"cross_case_edge:{edge.uid}")
            if edge.head_uid not in node_ids or edge.tail_uid not in node_ids:
                errors.append(f"dangling_edge:{edge.uid}")
            if edge.relation_type not in RELATIONS:
                errors.append(f"unknown_relation:{edge.uid}:{edge.relation_type}")
            source_documents = set(edge.source_document_ids or (edge.document_id,))
            if not source_documents.issubset(self.document_ids):
                errors.append(f"cross_case_edge_source:{edge.uid}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "document_ids": list(self.document_ids),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "source_mode": self.source_mode,
            "metadata": self.metadata,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def load_reconstructed_case_graphs(
    path: Path, source_graph_path: Path | None = None
) -> list[CaseGraph]:
    """Load validated reconstructed case graphs with source provenance."""
    source_texts: dict[str, str] = {}
    if source_graph_path is not None:
        source_texts = {
            str(row["document_id"]): str(row.get("text", ""))
            for row in _read_jsonl(source_graph_path)
        }
    result: list[CaseGraph] = []
    for row in _read_jsonl(path):
        case_id = str(row["case_cluster_id"])
        validation = dict(row.get("validation", {}))
        if not validation.get("passed", False):
            raise ValueError(f"Section 3.6 graph did not pass validation: {case_id}")
        document_ids = tuple(sorted(map(str, row.get("source_document_ids", []))))
        event_provenance = dict(row.get("provenance", {}).get("events", {}))
        relation_provenance = dict(row.get("provenance", {}).get("relations", {}))
        nodes: list[EventNode] = []
        for raw in row.get("events", []):
            uid = str(raw["case_event_id"])
            sources = list(event_provenance.get(uid, {}).get("sources", []))
            canonical_source_id = str(raw.get("canonical_source_event_id", ""))
            representative = next(
                (
                    source
                    for source in sources
                    if str(source.get("source_event_id")) == canonical_source_id
                ),
                sources[0] if sources else {},
            )
            document_id = str(
                representative.get("document_id", raw.get("source_document_ids", [""])[0])
            )
            span = representative.get("character_span", [-1, -1])
            start, end = int(span[0]), int(span[1])
            source_spans = tuple(
                (
                    int(source.get("character_span", [-1, -1])[0]),
                    int(source.get("character_span", [-1, -1])[1]),
                )
                for source in sources
            )
            span_checks: list[bool] = []
            for source in sources:
                source_document = str(source.get("document_id", ""))
                source_text = source_texts.get(source_document)
                source_span = source.get("character_span", [-1, -1])
                left, right = int(source_span[0]), int(source_span[1])
                mention = str(source.get("text", ""))
                span_checks.append(
                    source_text is not None
                    and 0 <= left < right <= len(source_text)
                    and source_text[left:right] == mention
                )
            score_values = [float(value) for value in raw.get("extraction_scores", [])]
            nodes.append(
                EventNode(
                    uid=uid,
                    case_id=case_id,
                    document_id=document_id,
                    event_id=uid,
                    text=str(raw.get("canonical_text", "")),
                    role=str(raw.get("event_type", "")),
                    start=start,
                    end=end,
                    score=max(score_values, default=0.0),
                    source_span_valid=bool(span_checks) and all(span_checks),
                    source_event_ids=tuple(map(str, raw.get("source_event_ids", []))),
                    source_document_ids=tuple(map(str, raw.get("source_document_ids", []))),
                    source_spans=source_spans,
                    canonical_source_event_id=canonical_source_id,
                )
            )
        edges: list[RelationEdge] = []
        for raw in row.get("relations", []):
            uid = str(raw["case_relation_id"])
            sources = list(relation_provenance.get(uid, {}).get("sources", []))
            source_documents = tuple(map(str, raw.get("source_document_ids", [])))
            document_id = (
                str(sources[0].get("document_id", source_documents[0] if source_documents else ""))
                if sources
                else (source_documents[0] if source_documents else "")
            )
            score_values = [float(value) for value in raw.get("extraction_scores", [])]
            evidence_values = [str(value) for value in raw.get("evidence_texts", []) if value]
            edges.append(
                RelationEdge(
                    uid=uid,
                    case_id=case_id,
                    document_id=document_id,
                    relation_id=uid,
                    head_uid=str(raw.get("head_case_event_id", "")),
                    tail_uid=str(raw.get("tail_case_event_id", "")),
                    relation_type=str(raw.get("relation_type", "")),
                    score=max(score_values, default=0.0),
                    evidence_text=max(evidence_values, key=len, default=""),
                    source_relation_ids=tuple(map(str, raw.get("source_relation_ids", []))),
                    source_document_ids=source_documents,
                    relation_state=str(raw.get("relation_state", "retained")),
                    complementary_with=tuple(map(str, raw.get("complementary_with", []))),
                    conflict_with=tuple(map(str, raw.get("conflict_with", []))),
                    conflict_dimensions=tuple(map(str, raw.get("conflict_dimensions", []))),
                )
            )
        graph = CaseGraph(
            case_id=case_id,
            document_ids=document_ids,
            nodes=tuple(nodes),
            edges=tuple(edges),
            source_mode="constraint_preserving_case_graph",
            metadata={
                "validation": validation,
                "review_item_count": len(row.get("review_queue", [])),
            },
        )
        errors = graph.validate()
        if errors:
            raise ValueError(f"invalid Section 3.6 case graph {case_id}: {errors[:10]}")
        result.append(graph)
    return result


def summarize_case_graphs(graphs: list[CaseGraph]) -> dict[str, Any]:
    nodes = [node for graph in graphs for node in graph.nodes]
    edges = [edge for graph in graphs for edge in graph.edges]
    source_event_count = sum(len(node.source_event_ids) or 1 for node in nodes)
    source_relation_count = sum(len(edge.source_relation_ids) or 1 for edge in edges)
    return {
        "case_count": len(graphs),
        "document_count": sum(len(graph.document_ids) for graph in graphs),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "source_event_count": source_event_count,
        "source_relation_count": source_relation_count,
        "event_reconstruction_ratio": len(nodes) / max(1, source_event_count),
        "relation_reconstruction_ratio": len(edges) / max(1, source_relation_count),
        "multi_document_case_count": sum(len(graph.document_ids) > 1 for graph in graphs),
        "source_span_provenance_rate": sum(node.source_span_valid for node in nodes)
        / max(1, len(nodes)),
        "case_boundary_violation_count": sum(bool(graph.validate()) for graph in graphs),
        "source_mode_counts": dict(sorted(Counter(graph.source_mode for graph in graphs).items())),
    }
