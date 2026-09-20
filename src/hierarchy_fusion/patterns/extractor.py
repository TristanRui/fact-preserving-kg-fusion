from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass

from hierarchy_fusion.data.case_graph import CaseGraph, EventNode, RelationEdge

ALLOWED_ROLE_PATHS = {
    ("PHENOMENON", "FAILURE"),
    ("PHENOMENON", "ROOT_CAUSE"),
    ("FAILURE", "ROOT_CAUSE"),
    ("ROOT_CAUSE", "ACTION"),
    ("ACTION", "VERIFICATION"),
    ("PHENOMENON", "ROOT_CAUSE", "ACTION"),
    ("FAILURE", "ROOT_CAUSE", "ACTION"),
    ("ROOT_CAUSE", "ACTION", "VERIFICATION"),
    ("PHENOMENON", "ROOT_CAUSE", "ACTION", "VERIFICATION"),
    ("PHENOMENON", "FAILURE", "ROOT_CAUSE", "ACTION"),
    ("FAILURE", "ROOT_CAUSE", "ACTION", "VERIFICATION"),
    ("PHENOMENON", "FAILURE", "ROOT_CAUSE", "ACTION", "VERIFICATION"),
}


def signature(roles: tuple[str, ...], concepts: tuple[str, ...], relations: tuple[str, ...]) -> str:
    return (
        ">".join(f"{role}:{concept}" for role, concept in zip(roles, concepts, strict=True))
        + "||"
        + "|".join(relations)
    )


def pattern_identifier(pattern_signature: str) -> str:
    """Return a stable identifier without replacing the auditable signature."""
    digest = hashlib.sha1(pattern_signature.encode("utf-8")).hexdigest()[:16]
    return f"PAT_{digest}"


@dataclass(frozen=True, slots=True)
class PatternInstance:
    signature: str
    case_id: str
    roles: tuple[str, ...]
    concepts: tuple[str, ...]
    relations: tuple[str, ...]
    source_node_uids: tuple[str, ...]
    source_edge_uids: tuple[str, ...]
    source_document_ids: tuple[str, ...]
    orientation_normalized: bool = True

    @property
    def pattern_id(self) -> str:
        return pattern_identifier(self.signature)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["pattern_id"] = self.pattern_id
        for key in (
            "roles",
            "concepts",
            "relations",
            "source_node_uids",
            "source_edge_uids",
            "source_document_ids",
        ):
            value[key] = list(value[key])
        return value


def diagnostic_arc(edge: RelationEdge) -> tuple[str, str]:
    """Return the diagnostic traversal direction for a stored MCPG relation."""

    if edge.relation_type in {"CAUSE", "TREAT", "VERIFY"}:
        return edge.tail_uid, edge.head_uid
    return edge.head_uid, edge.tail_uid


def extract_patterns(
    graphs: list[CaseGraph],
    node_to_concept: dict[str, str],
    *,
    max_path_edges: int = 4,
) -> tuple[list[PatternInstance], dict[str, list[PatternInstance]]]:
    all_patterns: list[PatternInstance] = []
    by_case: dict[str, list[PatternInstance]] = {}
    for graph in graphs:
        nodes = {node.uid: node for node in graph.nodes}
        adjacency: dict[str, list[tuple[str, RelationEdge]]] = defaultdict(list)
        for edge in graph.edges:
            source, target = diagnostic_arc(edge)
            if source in nodes and target in nodes:
                adjacency[source].append((target, edge))
        patterns: list[PatternInstance] = []
        seen: set[tuple] = set()

        def walk(
            node_path: list[str],
            edge_path: list[RelationEdge],
            *,
            current_graph: CaseGraph = graph,
            current_nodes: dict[str, EventNode] = nodes,
            current_adjacency: dict[str, list[tuple[str, RelationEdge]]] = adjacency,
            current_patterns: list[PatternInstance] = patterns,
            current_seen: set[tuple] = seen,
        ) -> None:
            roles = tuple(current_nodes[uid].role for uid in node_path)
            if edge_path and roles in ALLOWED_ROLE_PATHS:
                concepts = tuple(node_to_concept[uid] for uid in node_path)
                relations = tuple(edge.relation_type for edge in edge_path)
                key = (node_path[0], tuple(edge.uid for edge in edge_path), node_path[-1])
                if key not in current_seen:
                    current_seen.add(key)
                    current_patterns.append(
                        PatternInstance(
                            signature=signature(roles, concepts, relations),
                            case_id=current_graph.case_id,
                            roles=roles,
                            concepts=concepts,
                            relations=relations,
                            source_node_uids=tuple(node_path),
                            source_edge_uids=tuple(edge.uid for edge in edge_path),
                            source_document_ids=tuple(
                                dict.fromkeys(
                                    document_id
                                    for uid in node_path
                                    for document_id in (
                                        current_nodes[uid].source_document_ids
                                        or (current_nodes[uid].document_id,)
                                    )
                                )
                            ),
                        )
                    )
            if len(edge_path) >= max_path_edges:
                return
            for target, edge in current_adjacency.get(node_path[-1], []):
                if target in node_path:
                    continue
                next_roles = tuple(current_nodes[uid].role for uid in [*node_path, target])
                if not any(path[: len(next_roles)] == next_roles for path in ALLOWED_ROLE_PATHS):
                    continue
                walk([*node_path, target], [*edge_path, edge])

        valid_starts = {path[0] for path in ALLOWED_ROLE_PATHS}
        for uid, node in sorted(nodes.items()):
            if node.role in valid_starts:
                walk([uid], [])
        by_case[graph.case_id] = patterns
        all_patterns.extend(patterns)
    return all_patterns, by_case
