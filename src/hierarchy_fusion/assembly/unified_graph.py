from __future__ import annotations

from dataclasses import dataclass

from hierarchy_fusion.coarsening.hierarchy_builder import HierarchyResult
from hierarchy_fusion.data.case_graph import CaseGraph
from hierarchy_fusion.patterns.extractor import PatternInstance, pattern_identifier


@dataclass(slots=True)
class UnifiedHierarchicalDiagnosticGraph:
    case_facts: dict[str, dict]
    canonical_concepts: dict[str, dict]
    diagnostic_patterns: dict[str, dict]
    diagnostic_topics: dict[str, dict]
    edges: list[dict]
    root_unit_ids: tuple[str, ...]
    case_leaf_unit_ids: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "schema_version": "hierarchy_fusion_v1",
            "object_families": {
                "case_facts": self.case_facts,
                "canonical_concepts": self.canonical_concepts,
                "diagnostic_patterns": self.diagnostic_patterns,
                "diagnostic_topics": self.diagnostic_topics,
            },
            "edges": self.edges,
            "root_unit_ids": list(self.root_unit_ids),
            "case_leaf_unit_ids": self.case_leaf_unit_ids,
        }

    def trace_topic_pattern_to_sources(self, topic_id: str, pattern_id: str) -> dict:
        topic = self.diagnostic_topics.get(topic_id)
        pattern = self.diagnostic_patterns.get(pattern_id)
        if topic is None:
            raise KeyError(f"unknown topic: {topic_id}")
        if pattern is None:
            raise KeyError(f"unknown pattern: {pattern_id}")
        if pattern_id not in topic["pattern_refs"]:
            raise ValueError(f"topic {topic_id} does not reference {pattern_id}")
        instances = [
            instance
            for instance in pattern["instances"]
            if instance["case_id"] in topic["covered_case_ids"]
        ]
        if not instances:
            raise ValueError("topic pattern reference has no supporting case instance")
        return {
            "topic_id": topic_id,
            "pattern_id": pattern_id,
            "supporting_case_ids": sorted({instance["case_id"] for instance in instances}),
            "source_event_ids": sorted(
                {value for instance in instances for value in instance["source_event_ids"]}
            ),
            "source_relation_ids": sorted(
                {value for instance in instances for value in instance["source_relation_ids"]}
            ),
            "source_document_ids": sorted(
                {value for instance in instances for value in instance["source_document_ids"]}
            ),
            "source_spans": sorted(
                {
                    (span["document_id"], span["start"], span["end"])
                    for instance in instances
                    for span in instance["source_spans"]
                }
            ),
        }


def _pattern_catalog(
    patterns_by_case: dict[str, list[PatternInstance]],
    graphs_by_case: dict[str, CaseGraph],
) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for case_id in sorted(patterns_by_case):
        graph = graphs_by_case[case_id]
        nodes = {node.uid: node for node in graph.nodes}
        edges = {edge.uid: edge for edge in graph.edges}
        for pattern in patterns_by_case[case_id]:
            pattern_id = pattern_identifier(pattern.signature)
            record = catalog.setdefault(
                pattern_id,
                {
                    "pattern_id": pattern_id,
                    "signature": pattern.signature,
                    "roles": list(pattern.roles),
                    "concept_ids": list(pattern.concepts),
                    "relation_types": list(pattern.relations),
                    "support_case_ids": [],
                    "instances": [],
                },
            )
            if case_id not in record["support_case_ids"]:
                record["support_case_ids"].append(case_id)
            source_event_ids = sorted(
                {
                    source_id
                    for uid in pattern.source_node_uids
                    for source_id in nodes[uid].source_event_ids
                }
            )
            source_relation_ids = sorted(
                {
                    source_id
                    for uid in pattern.source_edge_uids
                    for source_id in edges[uid].source_relation_ids
                }
            )
            source_spans = [
                {
                    "document_id": document_id,
                    "start": int(span[0]),
                    "end": int(span[1]),
                }
                for uid in pattern.source_node_uids
                for document_id, span in zip(
                    nodes[uid].source_document_ids,
                    nodes[uid].source_spans,
                    strict=True,
                )
            ]
            record["instances"].append(
                {
                    "case_id": case_id,
                    "case_event_ids": list(pattern.source_node_uids),
                    "case_relation_ids": list(pattern.source_edge_uids),
                    "source_event_ids": source_event_ids,
                    "source_relation_ids": source_relation_ids,
                    "source_document_ids": list(pattern.source_document_ids),
                    "source_spans": source_spans,
                }
            )
    for record in catalog.values():
        record["support_case_ids"].sort()
        record["case_support_count"] = len(record["support_case_ids"])
    return dict(sorted(catalog.items()))


def assemble_unified_graph(
    graphs: list[CaseGraph],
    canonical_concepts: dict[str, dict],
    node_to_concept: dict[str, str],
    patterns_by_case: dict[str, list[PatternInstance]],
    hierarchy: HierarchyResult,
    *,
    include_all_topics: bool = True,
) -> UnifiedHierarchicalDiagnosticGraph:
    graphs_by_case = {graph.case_id: graph for graph in graphs}
    case_facts = {
        graph.case_id: {
            "case_id": graph.case_id,
            "document_ids": list(graph.document_ids),
            "event_ids": [node.uid for node in graph.nodes],
            "relation_ids": [edge.uid for edge in graph.edges],
            "events": [node.to_dict() for node in graph.nodes],
            "relations": [edge.to_dict() for edge in graph.edges],
            "source_mode": graph.source_mode,
        }
        for graph in graphs
    }
    patterns = _pattern_catalog(patterns_by_case, graphs_by_case)
    topics = {
        unit_id: unit.to_dict(include_vector=True)
        for unit_id, unit in hierarchy.units.items()
        if unit.unit_type == "topic" and (include_all_topics or unit_id in hierarchy.root_ids)
    }
    case_leaf_unit_ids = {
        next(iter(unit.case_ids)): unit_id
        for unit_id, unit in hierarchy.units.items()
        if unit.unit_type == "case"
    }
    edges: list[dict] = []

    def add(namespace: str, edge_type: str, source: str, target: str, **metadata) -> None:
        edges.append(
            {
                "namespace": namespace,
                "edge_type": edge_type,
                "source_id": source,
                "target_id": target,
                **metadata,
            }
        )

    for graph in graphs:
        for relation in graph.edges:
            add(
                "factual",
                relation.relation_type,
                relation.head_uid,
                relation.tail_uid,
                case_id=graph.case_id,
                case_relation_id=relation.uid,
            )
        for node in graph.nodes:
            add(
                "reference",
                "EVENT_TO_CANONICAL_CONCEPT",
                node.uid,
                node_to_concept[node.uid],
                case_id=graph.case_id,
            )

    for pattern_id, pattern in patterns.items():
        for concept_id in sorted(set(pattern["concept_ids"])):
            add("reference", "PATTERN_TO_CONCEPT", pattern_id, concept_id)
        for case_id in pattern["support_case_ids"]:
            add("reference", "PATTERN_TO_SUPPORTING_CASE", pattern_id, case_id)

    for topic_id, topic in topics.items():
        for concept_id in topic["canonical_concept_ids"]:
            add("reference", "TOPIC_TO_CONCEPT", topic_id, concept_id)
        for pattern_id in topic["pattern_refs"]:
            add("reference", "TOPIC_TO_PATTERN", topic_id, pattern_id)
        for case_id in topic["covered_case_ids"]:
            add("reference", "TOPIC_TO_COVERED_CASE", topic_id, case_id)
        for child_id in topic["child_unit_ids"]:
            child = hierarchy.units[child_id]
            if child.unit_type == "case":
                add(
                    "hierarchical",
                    "TOPIC_TO_CHILD_CASE",
                    topic_id,
                    next(iter(child.case_ids)),
                    child_unit_id=child_id,
                )
            elif child_id in topics:
                add("hierarchical", "TOPIC_TO_CHILD_TOPIC", topic_id, child_id)

    return UnifiedHierarchicalDiagnosticGraph(
        case_facts=dict(sorted(case_facts.items())),
        canonical_concepts=dict(sorted(canonical_concepts.items())),
        diagnostic_patterns=patterns,
        diagnostic_topics=dict(sorted(topics.items())),
        edges=edges,
        root_unit_ids=hierarchy.root_ids,
        case_leaf_unit_ids=case_leaf_unit_ids,
    )
