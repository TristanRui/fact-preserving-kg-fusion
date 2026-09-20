"""Conservative structural-backbone construction for hierarchical fusion."""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np

from hierarchy_fusion.coarsening.candidate_graph import build_ann_candidate_pairs
from hierarchy_fusion.coarsening.hierarchy_builder import (
    DiagnosticUnit,
    HierarchyBuilder,
    HierarchyResult,
    normalize,
)


def _association_graph(
    builder: HierarchyBuilder, *, candidate_top_k: int, threshold: float
) -> nx.Graph:
    case_ids = sorted(builder.case_vectors)
    units, case_to_leaf = builder._leaf_units()
    graph = nx.Graph()
    graph.add_nodes_from(case_ids)
    for left_case, right_case in build_ann_candidate_pairs(
        case_ids, builder.case_vectors, top_k=candidate_top_k
    ):
        score = builder.score_units(
            units[case_to_leaf[left_case]], units[case_to_leaf[right_case]]
        )["S_assoc"]
        if score >= threshold:
            graph.add_edge(left_case, right_case, weight=float(score))
    return graph


def _connected_chunks(graph: nx.Graph, nodes: set[str], maximum_size: int) -> list[set[str]]:
    remaining = set(nodes)
    chunks: list[set[str]] = []
    subgraph = graph.subgraph(nodes)
    while remaining:
        seed = sorted(remaining, key=lambda node: (-subgraph.degree(node), node))[0]
        chunk = {seed}
        frontier = set(subgraph.neighbors(seed)) & remaining
        while frontier and len(chunk) < maximum_size:
            choice = sorted(
                frontier,
                key=lambda node: (
                    -sum(subgraph.has_edge(node, member) for member in chunk),
                    node,
                ),
            )[0]
            chunk.add(choice)
            frontier = (frontier | (set(subgraph.neighbors(choice)) & remaining)) - chunk
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def _materialize_partitions(
    builder: HierarchyBuilder, partitions: list[set[str]]
) -> HierarchyResult:
    units, case_to_leaf = builder._leaf_units()
    covered = set().union(*partitions) if partitions else set()
    if covered != set(case_to_leaf) or sum(map(len, partitions)) != len(covered):
        raise ValueError("structural partitions must cover every case exactly once")
    roots: list[str] = []
    trace: list[dict] = []
    accepted_merges = 0
    for topic_index, partition in enumerate(sorted(partitions, key=lambda values: min(values))):
        child_ids = [case_to_leaf[case_id] for case_id in sorted(partition)]
        if len(child_ids) == 1:
            roots.append(child_ids[0])
            continue
        topic_id = f"B{topic_index:04d}"
        case_ids = frozenset(partition)
        leaf_sum = np.sum([builder.case_vectors[case_id] for case_id in sorted(case_ids)], axis=0)
        pattern_refs = frozenset().union(
            *(builder.case_pattern_signatures[case_id] for case_id in case_ids)
        )
        units[topic_id] = DiagnosticUnit(
            unit_id=topic_id,
            unit_type="topic",
            case_ids=case_ids,
            child_ids=tuple(child_ids),
            level=1,
            vector=normalize(leaf_sum),
            leaf_vector_sum=leaf_sum,
            concept_ids=frozenset().union(
                *(builder.case_concepts[case_id] for case_id in case_ids)
            ),
            evidence_features=builder.evidence_index.topic_features(case_ids),
            pattern_refs=pattern_refs,
            pattern_support_case_ids=builder._pattern_support(pattern_refs, case_ids),
        )
        for child_id in child_ids:
            units[child_id].active = False
            units[child_id].parent_topic_id = topic_id
        for child_id in child_ids[1:]:
            accepted_merges += 1
            trace.append(
                {
                    "iteration": accepted_merges,
                    "unit_a": child_ids[0],
                    "unit_b": child_id,
                    "covered_case_ids_a": sorted(units[child_ids[0]].case_ids),
                    "covered_case_ids_b": sorted(units[child_id].case_ids),
                    "accepted": True,
                    "parent_topic_id": topic_id,
                    "reason": "structural_backbone_partition",
                }
            )
        roots.append(topic_id)
    return HierarchyResult(
        variant="structural_backbone",
        units=units,
        root_ids=tuple(sorted(roots)),
        association_trace=trace,
        candidate_pair_count=accepted_merges,
        accepted_merge_count=accepted_merges,
        termination_reason="structural_backbone_complete",
        config=builder.config,
    )


def build_structural_backbone(
    builder: HierarchyBuilder,
    *,
    candidate_top_k: int,
    threshold: float,
    maximum_cluster_size: int,
) -> HierarchyResult:
    """Build the paper's k-core shell backbone with bounded connected chunks."""

    graph = _association_graph(builder, candidate_top_k=candidate_top_k, threshold=threshold)
    core_numbers = nx.core_number(graph) if graph.number_of_edges() else {node: 0 for node in graph}
    partitions: list[set[str]] = []
    for component in nx.connected_components(graph):
        pending: deque[set[str]] = deque([set(component)])
        while pending:
            group = pending.popleft()
            if len(group) <= maximum_cluster_size:
                partitions.append(group)
                continue
            shell_values = sorted({core_numbers[node] for node in group})
            pivot = shell_values[len(shell_values) // 2]
            dense = {node for node in group if core_numbers[node] > pivot}
            sparse = group - dense
            pieces: list[set[str]] = []
            for subset in (dense, sparse):
                if subset:
                    pieces.extend(
                        set(part) for part in nx.connected_components(graph.subgraph(subset))
                    )
            if len(pieces) == 1 and pieces[0] == group:
                partitions.extend(_connected_chunks(graph, group, maximum_cluster_size))
            else:
                pending.extend(sorted(pieces, key=lambda values: min(values)))
    return _materialize_partitions(builder, partitions)
