"""Load immutable case graphs and construct the final hierarchy inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hierarchy_fusion.coarsening.hierarchy_builder import AssociationConfig, HierarchyBuilder
from hierarchy_fusion.coarsening.similarity import (
    FixedLeafEvidenceIndex,
    IdfWeightedDirectionAwareEvidenceIndex,
)
from hierarchy_fusion.data.case_graph import load_reconstructed_case_graphs
from hierarchy_fusion.evidence.closure import EvidenceClosureIndex
from hierarchy_fusion.patterns.extractor import extract_patterns
from hierarchy_fusion.semantics.canonicalizer import canonicalize
from hierarchy_fusion.semantics.embedding import (
    load_case_bge_embeddings,
    load_event_context_bge_embeddings,
)


def prepare_hierarchy_inputs(project_root: Path, config_path: Path) -> dict[str, Any]:
    """Prepare label-free, leaf-grounded inputs for hierarchical fusion."""

    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hierarchy = configuration["hierarchy"]
    inputs = hierarchy["input"]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (project_root / path).resolve()

    graph_path = resolve(inputs["case_graph_path"])
    source_graph_path = resolve(inputs["source_graph_path"])
    graphs = load_reconstructed_case_graphs(graph_path, source_graph_path)

    event_vectors = None
    event_embedding_path = inputs.get("event_embedding_path")
    if event_embedding_path:
        event_vectors, _ = load_event_context_bge_embeddings(
            graphs,
            source_graph_path,
            resolve(event_embedding_path),
        )
    canonicalization = hierarchy["canonicalization"]
    canonical = canonicalize(
        graphs,
        threshold=float(canonicalization["lexical_similarity_threshold"]),
        top_k=int(canonicalization["neighbor_top_k"]),
        ngram_range=(
            int(canonicalization["char_ngram_min"]),
            int(canonicalization["char_ngram_max"]),
        ),
        max_features=int(canonicalization["max_features"]),
        semantic_vectors=event_vectors,
        semantic_threshold=float(canonicalization["context_bge_similarity_threshold"]),
        semantic_top_k=int(canonicalization["context_bge_neighbor_top_k"]),
    )
    _, patterns_by_case = extract_patterns(
        graphs,
        canonical.node_to_concept,
        max_path_edges=int(hierarchy["patterns"]["max_path_edges"]),
    )
    evidence_index = FixedLeafEvidenceIndex(graphs, canonical.node_to_concept, patterns_by_case)
    candidate_evidence = IdfWeightedDirectionAwareEvidenceIndex(patterns_by_case)
    closure = EvidenceClosureIndex(patterns_by_case)
    case_vectors, _, _ = load_case_bge_embeddings(
        graphs,
        resolve(inputs["document_embedding_path"]),
        resolve(inputs["document_embedding_manifest"])
        if inputs.get("document_embedding_manifest")
        else None,
    )
    case_concepts = {
        graph.case_id: {canonical.node_to_concept[node.uid] for node in graph.nodes}
        for graph in graphs
    }
    association = hierarchy["association"]
    builder = HierarchyBuilder(
        case_vectors=case_vectors,
        case_concepts=case_concepts,
        patterns_by_case=patterns_by_case,
        evidence_index=evidence_index,
        config=AssociationConfig(
            alpha=float(association["semantic_weight"]),
            beta=float(association["evidence_weight"]),
            tau_assoc=float(association["threshold"]),
            candidate_top_k=int(hierarchy["candidate_top_k"]),
            max_merges=max(0, len(graphs) - 1),
            allow_multiple_roots=True,
        ),
    )
    return {
        "configuration": configuration,
        "graphs": graphs,
        "canonical": canonical,
        "patterns_by_case": patterns_by_case,
        "evidence_index": evidence_index,
        "candidate_evidence": candidate_evidence,
        "closure": closure,
        "case_vectors": case_vectors,
        "case_concepts": case_concepts,
        "builder": builder,
    }
