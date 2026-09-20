"""Build the manuscript's structure-first non-transitive hierarchy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierarchy_fusion.assembly.unified_graph import assemble_unified_graph
from hierarchy_fusion.coarsening.route_b_hierarchy import (
    RouteBHierarchyConfig,
    build_route_b_hierarchy,
)
from hierarchy_fusion.fusion.role_aware_sparse import (
    RoleAwareSparseIndex,
    RoleAwareSparseResidualConfig,
)
from hierarchy_fusion.pipeline import prepare_hierarchy_inputs


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.config.resolve().parents[1]
    assets = prepare_hierarchy_inputs(project_root, args.config.resolve())
    hierarchy_config = assets["configuration"]["hierarchy"]
    residual_config = hierarchy_config["residual"]
    residual = RoleAwareSparseResidualConfig(
        role=str(residual_config["role"]),
        minimum_similarity=float(residual_config["minimum_similarity"]),
        allowed_candidate_types=frozenset(residual_config["allowed_candidate_types"]),
    )
    hierarchy = build_route_b_hierarchy(
        assets["builder"],
        closure=assets["closure"],
        role_index=RoleAwareSparseIndex.from_case_graphs(assets["graphs"], role=residual.role),
        candidate_evidence=assets["candidate_evidence"],
        config=RouteBHierarchyConfig(
            candidate_top_k=int(hierarchy_config["candidate_top_k"]),
            backbone_threshold=float(hierarchy_config["association"]["threshold"]),
            backbone_max_cluster_size=int(hierarchy_config["backbone_max_cluster_size"]),
            residual=residual,
        ),
    )
    unified = assemble_unified_graph(
        assets["graphs"],
        assets["canonical"].concepts,
        assets["canonical"].node_to_concept,
        assets["patterns_by_case"],
        hierarchy,
        include_all_topics=True,
    )
    write_json(args.output_dir / "summary.json", hierarchy.summary())
    write_jsonl(
        args.output_dir / "hierarchy_units.jsonl",
        [hierarchy.units[unit_id].to_dict() for unit_id in sorted(hierarchy.units)],
    )
    write_jsonl(args.output_dir / "association_trace.jsonl", hierarchy.association_trace)
    write_json(args.output_dir / "hierarchical_graph.json", unified.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
