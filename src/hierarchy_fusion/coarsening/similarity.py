from __future__ import annotations

import math
from collections import Counter

from hierarchy_fusion.data.case_graph import CaseGraph
from hierarchy_fusion.patterns.extractor import PatternInstance, diagnostic_arc


def pattern_features(pattern: PatternInstance) -> set[str]:
    features = {
        "path:" + ">".join(pattern.roles) + "||" + "|".join(pattern.relations),
    }
    for idx, relation in enumerate(pattern.relations):
        features.add(f"arc:{pattern.roles[idx]}>{relation}>{pattern.roles[idx + 1]}")
    features.update(
        f"concept:{role}:{concept}"
        for role, concept in zip(pattern.roles, pattern.concepts, strict=True)
    )
    return features


class IdfWeightedDirectionAwareEvidenceIndex:
    """Symmetric IDF-weighted similarity over direction-aware evidence features."""

    def __init__(self, patterns_by_case: dict[str, list[PatternInstance]]) -> None:
        self.case_features: dict[str, frozenset[str]] = {}
        df: Counter[str] = Counter()
        for case_id, patterns in patterns_by_case.items():
            features = (
                frozenset().union(*(pattern_features(pattern) for pattern in patterns))
                if patterns
                else frozenset()
            )
            self.case_features[case_id] = features
            df.update(features)
        total = max(1, len(patterns_by_case))
        self.idf = {
            feature: math.log((total + 1) / (count + 1)) + 1.0 for feature, count in df.items()
        }

    def cluster_features(self, case_ids: set[str]) -> frozenset[str]:
        return (
            frozenset().union(
                *(self.case_features.get(case_id, frozenset()) for case_id in case_ids)
            )
            if case_ids
            else frozenset()
        )

    def weighted_jaccard(self, left: frozenset[str], right: frozenset[str]) -> float:
        union = left | right
        if not union:
            return 0.0
        # Sort before floating-point reduction so identical inputs produce
        # byte-identical audit artifacts across Python hash seeds/processes.
        numerator = math.fsum(self.idf.get(feature, 1.0) for feature in sorted(left & right))
        denominator = math.fsum(self.idf.get(feature, 1.0) for feature in sorted(union))
        return numerator / max(denominator, 1e-12)


# Backward-compatible internal name. Manuscript terminology must use
# "IDF-weighted direction-aware evidence similarity": the direction is encoded
# in each feature, while the weighted-Jaccard comparison itself is symmetric.
DirectedIdfEvidenceIndex = IdfWeightedDirectionAwareEvidenceIndex


def case_evidence_features(
    graph: CaseGraph,
    node_to_concept: dict[str, str],
    patterns: list[PatternInstance],
    *,
    feature_families: frozenset[str] = frozenset({"role_concept", "arc", "pattern"}),
) -> frozenset[str]:
    """Build the manuscript-defined leaf evidence feature set.

    Feature identities retain diagnostic role, direction and configuration.
    They are references over immutable case facts, never replacements for the
    original event or relation objects.
    """
    nodes = {node.uid: node for node in graph.nodes}
    allowed = {"role_concept", "arc", "pattern"}
    unknown = set(feature_families) - allowed
    if unknown:
        raise ValueError(f"unknown evidence feature families: {sorted(unknown)}")
    features: set[str] = set()
    if "role_concept" in feature_families:
        features.update(
            f"role_concept:{node.role}:{node_to_concept[node.uid]}" for node in graph.nodes
        )
    if "arc" in feature_families:
        for edge in graph.edges:
            source_uid, target_uid = diagnostic_arc(edge)
            if source_uid not in nodes or target_uid not in nodes:
                continue
            source = nodes[source_uid]
            target = nodes[target_uid]
            features.add(
                "arc:"
                f"{source.role}:{node_to_concept[source_uid]}"
                f">{edge.relation_type}>"
                f"{target.role}:{node_to_concept[target_uid]}"
            )
    if "pattern" in feature_families:
        features.update(f"pattern:{pattern.signature}" for pattern in patterns)
    return frozenset(features)


class FixedLeafEvidenceIndex:
    """Fixed leaf-corpus IDF and weighted-Jaccard evidence similarity."""

    def __init__(
        self,
        graphs: list[CaseGraph],
        node_to_concept: dict[str, str],
        patterns_by_case: dict[str, list[PatternInstance]],
        *,
        feature_families: frozenset[str] = frozenset({"role_concept", "arc", "pattern"}),
        reference_idf: dict[str, float] | None = None,
        unseen_idf: float | None = None,
    ) -> None:
        self.feature_families = frozenset(feature_families)
        self.case_features: dict[str, frozenset[str]] = {}
        document_frequency: Counter[str] = Counter()
        for graph in graphs:
            features = case_evidence_features(
                graph,
                node_to_concept,
                patterns_by_case.get(graph.case_id, []),
                feature_families=self.feature_families,
            )
            self.case_features[graph.case_id] = features
            document_frequency.update(features)
        self.leaf_case_count = len(graphs)
        total = max(1, self.leaf_case_count)
        if reference_idf is None:
            self.idf = {
                feature: math.log((total + 1) / (count + 1)) + 1.0
                for feature, count in document_frequency.items()
            }
        else:
            fallback = float(
                unseen_idf if unseen_idf is not None else max(reference_idf.values(), default=1.0)
            )
            self.idf = {
                feature: float(reference_idf.get(feature, fallback))
                for feature in document_frequency
            }

    def topic_features(self, case_ids: set[str] | frozenset[str]) -> frozenset[str]:
        """Deterministic evidence-preserving union over covered leaf cases."""
        if not case_ids:
            return frozenset()
        return frozenset().union(
            *(self.case_features.get(case_id, frozenset()) for case_id in sorted(case_ids))
        )

    def weighted_jaccard(
        self,
        left: frozenset[str],
        right: frozenset[str],
    ) -> float:
        union = left | right
        if not union:
            return 0.0
        unknown = union - self.idf.keys()
        if unknown:
            raise ValueError(
                "topic evidence contains features outside the frozen leaf corpus: "
                f"{sorted(unknown)[:3]}"
            )
        numerator = math.fsum(self.idf[feature] for feature in sorted(left & right))
        denominator = math.fsum(self.idf[feature] for feature in sorted(union))
        return numerator / max(denominator, 1e-12)


def association_score(
    semantic_similarity: float,
    evidence_similarity: float,
    *,
    alpha: float,
    beta: float,
) -> float:
    """S_assoc = alpha * S_sem + beta * S_evi, with no hidden term."""
    return alpha * semantic_similarity + beta * evidence_similarity
