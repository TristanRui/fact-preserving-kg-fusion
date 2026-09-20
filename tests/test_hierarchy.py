import numpy as np

from hierarchy_fusion.coarsening.hierarchy_builder import AssociationConfig, HierarchyBuilder
from hierarchy_fusion.coarsening.route_b_hierarchy import (
    RouteBHierarchyConfig,
    build_route_b_hierarchy,
)
from hierarchy_fusion.coarsening.similarity import (
    FixedLeafEvidenceIndex,
    IdfWeightedDirectionAwareEvidenceIndex,
)
from hierarchy_fusion.data.case_graph import CaseGraph, EventNode
from hierarchy_fusion.evidence.closure import EvidenceClosureIndex
from hierarchy_fusion.fusion.role_aware_sparse import (
    RoleAwareSparseIndex,
    RoleAwareSparseResidualConfig,
)


def graph(case_id: str) -> CaseGraph:
    node = EventNode(
        uid=f"{case_id}:E1",
        case_id=case_id,
        document_id=f"{case_id}:D1",
        event_id="E1",
        text="密封失效",
        role="ROOT_CAUSE",
        start=0,
        end=4,
        score=1.0,
        source_span_valid=True,
        source_event_ids=(f"{case_id}:D1_E1",),
        source_document_ids=(f"{case_id}:D1",),
        source_spans=((0, 4),),
        canonical_source_event_id=f"{case_id}:D1_E1",
    )
    return CaseGraph(
        case_id=case_id,
        document_ids=(f"{case_id}:D1",),
        nodes=(node,),
        edges=(),
        source_mode="test",
    )


def test_residual_matching_is_non_transitive() -> None:
    graphs = [graph("A"), graph("B"), graph("C")]
    node_to_concept = {item.nodes[0].uid: "ROOT_CAUSE:seal" for item in graphs}
    patterns_by_case = {item.case_id: [] for item in graphs}
    evidence = FixedLeafEvidenceIndex(graphs, node_to_concept, patterns_by_case)
    builder = HierarchyBuilder(
        case_vectors={case_id: np.array([1.0, 0.0]) for case_id in patterns_by_case},
        case_concepts={case_id: {"ROOT_CAUSE:seal"} for case_id in patterns_by_case},
        patterns_by_case=patterns_by_case,
        evidence_index=evidence,
        config=AssociationConfig(
            alpha=0.22,
            beta=0.78,
            tau_assoc=0.9,
            candidate_top_k=2,
            allow_multiple_roots=True,
        ),
    )
    result = build_route_b_hierarchy(
        builder,
        closure=EvidenceClosureIndex(patterns_by_case),
        role_index=RoleAwareSparseIndex.from_case_graphs(graphs, role="ROOT_CAUSE"),
        candidate_evidence=IdfWeightedDirectionAwareEvidenceIndex(patterns_by_case),
        config=RouteBHierarchyConfig(
            candidate_top_k=2,
            backbone_threshold=1.1,
            backbone_max_cluster_size=20,
            residual=RoleAwareSparseResidualConfig(minimum_similarity=0.01),
        ),
    )
    roots = [result.units[root_id] for root_id in result.root_ids]
    assert len(roots) == 2
    assert max(len(root.case_ids) for root in roots) == 2
    assert max(root.level for root in roots) == 1
