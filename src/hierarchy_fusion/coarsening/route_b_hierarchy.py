"""Structure-first, non-transitive hierarchical diagnostic graph fusion."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from hierarchy_fusion.coarsening.candidate_graph import build_ann_candidate_pairs
from hierarchy_fusion.coarsening.hierarchy_builder import (
    DiagnosticUnit,
    HierarchyBuilder,
    HierarchyResult,
    normalize,
)
from hierarchy_fusion.coarsening.risk import classify_candidate_risk
from hierarchy_fusion.coarsening.similarity import IdfWeightedDirectionAwareEvidenceIndex
from hierarchy_fusion.coarsening.structural_backbone import build_structural_backbone
from hierarchy_fusion.evidence.closure import EvidenceClosureIndex
from hierarchy_fusion.fusion.role_aware_sparse import (
    RoleAwareSparseIndex,
    RoleAwareSparseResidualConfig,
)


@dataclass(frozen=True, slots=True)
class RouteBHierarchyConfig:
    """Frozen parameters reported for the manuscript's final hierarchy."""

    candidate_top_k: int = 12
    backbone_threshold: float = 0.19694262623786926
    backbone_max_cluster_size: int = 20
    residual: RoleAwareSparseResidualConfig = RoleAwareSparseResidualConfig()


def build_route_b_hierarchy(
    builder: HierarchyBuilder,
    *,
    closure: EvidenceClosureIndex,
    role_index: RoleAwareSparseIndex,
    candidate_evidence: IdfWeightedDirectionAwareEvidenceIndex,
    config: RouteBHierarchyConfig | None = None,
) -> HierarchyResult:
    """Build the bounded backbone and one non-recursive residual layer.

    Each accepted root pair retains its strongest leaf-case witness. Maximum-
    weight matching enforces degree at most one, so newly created topics never
    re-enter residual candidate generation.
    """

    active = config or RouteBHierarchyConfig()
    backbone = build_structural_backbone(
        builder,
        candidate_top_k=active.candidate_top_k,
        threshold=active.backbone_threshold,
        maximum_cluster_size=active.backbone_max_cluster_size,
    )
    units = backbone.units
    root_by_case = {
        case_id: root_id for root_id in backbone.root_ids for case_id in units[root_id].case_ids
    }
    leaf_candidates = build_ann_candidate_pairs(
        sorted(builder.case_vectors),
        builder.case_vectors,
        top_k=active.candidate_top_k,
    )
    strongest_witness: dict[tuple[str, str], dict] = {}
    for left_case, right_case in sorted(leaf_candidates):
        left_root = root_by_case[left_case]
        right_root = root_by_case[right_case]
        if left_root == right_root:
            continue
        root_pair = tuple(sorted((left_root, right_root)))
        assessment = closure.assess({left_case, right_case})
        semantic_similarity = max(
            0.0,
            float(np.dot(builder.case_vectors[left_case], builder.case_vectors[right_case])),
        )
        evidence_similarity = candidate_evidence.weighted_jaccard(
            candidate_evidence.case_features[left_case],
            candidate_evidence.case_features[right_case],
        )
        shared_concepts = set(builder.case_concepts[left_case]) & set(
            builder.case_concepts[right_case]
        )
        risk = classify_candidate_risk(
            unsupported_patterns=assessment.unsupported_signatures,
            shared_concepts=shared_concepts,
            semantic_similarity=semantic_similarity,
            evidence_similarity=evidence_similarity,
        )
        role_similarity = role_index.similarity(left_case, right_case)
        accepted = (
            risk in active.residual.allowed_candidate_types
            and role_similarity >= active.residual.minimum_similarity
        )
        record = {
            "left_root_id": root_pair[0],
            "right_root_id": root_pair[1],
            "witness_left_case_id": left_case,
            "witness_right_case_id": right_case,
            "candidate_type": risk,
            "semantic_similarity": semantic_similarity,
            "direction_aware_evidence_similarity": evidence_similarity,
            "role_similarity": role_similarity,
            "minimum_role_similarity": active.residual.minimum_similarity,
            "allowed_candidate_type": risk in active.residual.allowed_candidate_types,
            "accepted_by_residual_rule": accepted,
            "unsupported_pattern_count": len(assessment.unsupported_signatures),
        }
        previous = strongest_witness.get(root_pair)
        record_rank = (accepted, role_similarity, left_case, right_case)
        previous_rank = (
            (
                previous["accepted_by_residual_rule"],
                previous["role_similarity"],
                previous["witness_left_case_id"],
                previous["witness_right_case_id"],
            )
            if previous
            else None
        )
        if previous is None or record_rank > previous_rank:
            strongest_witness[root_pair] = record

    residual_graph = nx.Graph()
    residual_graph.add_nodes_from(backbone.root_ids)
    for (left_root, right_root), record in strongest_witness.items():
        if record["accepted_by_residual_rule"]:
            residual_graph.add_edge(
                left_root,
                right_root,
                weight=float(record["role_similarity"]),
            )
    matching = nx.algorithms.matching.max_weight_matching(
        residual_graph, maxcardinality=False, weight="weight"
    )
    matched_pairs = {tuple(sorted(pair)) for pair in matching}

    roots = set(backbone.root_ids)
    residual_trace: list[dict] = []
    for iteration, (root_pair, record) in enumerate(sorted(strongest_witness.items()), start=1):
        selected = root_pair in matched_pairs
        residual_trace.append(
            {
                "stage": "root_cause_residual",
                "iteration": iteration,
                "unit_a": root_pair[0],
                "unit_b": root_pair[1],
                "covered_case_ids_a": sorted(units[root_pair[0]].case_ids),
                "covered_case_ids_b": sorted(units[root_pair[1]].case_ids),
                **record,
                "accepted": selected,
                "reason": (
                    "selected_by_non_transitive_maximum_weight_matching"
                    if selected
                    else (
                        "eligible_but_not_selected_by_matching"
                        if record["accepted_by_residual_rule"]
                        else "rejected_by_typed_residual_rule"
                    )
                ),
            }
        )

    for topic_index, (left_root, right_root) in enumerate(sorted(matched_pairs)):
        left = units[left_root]
        right = units[right_root]
        case_ids = left.case_ids | right.case_ids
        topic_id = f"R{topic_index:04d}"
        leaf_sum = np.sum([builder.case_vectors[case_id] for case_id in sorted(case_ids)], axis=0)
        pattern_refs = left.pattern_refs | right.pattern_refs
        units[topic_id] = DiagnosticUnit(
            unit_id=topic_id,
            unit_type="topic",
            case_ids=case_ids,
            child_ids=(left_root, right_root),
            level=max(left.level, right.level) + 1,
            vector=normalize(leaf_sum),
            leaf_vector_sum=leaf_sum,
            concept_ids=left.concept_ids | right.concept_ids,
            evidence_features=builder.evidence_index.topic_features(case_ids),
            pattern_refs=pattern_refs,
            pattern_support_case_ids=builder._pattern_support(pattern_refs, case_ids),
        )
        left.active = False
        right.active = False
        left.parent_topic_id = topic_id
        right.parent_topic_id = topic_id
        roots.discard(left_root)
        roots.discard(right_root)
        roots.add(topic_id)

    trace = [
        {"stage": "structural_backbone", **record} for record in backbone.association_trace
    ] + residual_trace
    return HierarchyResult(
        variant="structure_first_non_transitive_hierarchy",
        units=units,
        root_ids=tuple(sorted(roots)),
        association_trace=trace,
        candidate_pair_count=len(leaf_candidates),
        accepted_merge_count=backbone.accepted_merge_count + len(matched_pairs),
        termination_reason="backbone_and_single_residual_matching_complete",
        config=builder.config,
    )
