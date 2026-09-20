"""Structural-risk categories used by residual cross-case association."""

from __future__ import annotations


def _node_count(pattern_signature: str) -> int:
    path = pattern_signature.split("||", 1)[0]
    return path.count(">") + 1 if path else 0


def classify_candidate_risk(
    *,
    unsupported_patterns: frozenset[str],
    shared_concepts: set[str],
    semantic_similarity: float,
    evidence_similarity: float,
) -> str:
    """Classify a candidate using the ordered risk rules from Section 4.3."""

    if (
        unsupported_patterns
        and max((_node_count(value) for value in unsupported_patterns), default=0) >= 4
    ):
        return "E_higher_order_cross_case_chain"
    if unsupported_patterns and shared_concepts:
        return "C_shared_anchor_path_divergence"
    if unsupported_patterns:
        return "A_local_edges_compose_without_case_support"
    if semantic_similarity >= 0.80 and evidence_similarity <= 0.05:
        return "D_bge_high_structure_low"
    return "safe_or_low_risk_control"
