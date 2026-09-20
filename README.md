# Fact-Preserving Knowledge Graph Fusion

Reference implementation of *Fact-preserving hierarchical fusion of multi-source maintenance case knowledge graphs for aero-engine anomaly diagnosis*.

The repository contains only the method pipeline. Private datasets, annotations, trained artifacts, generated results, comparison methods, ablation code, and manuscript figures are intentionally excluded.

## Method pipeline

1. Resolve maintenance records into case identities with `0.75 × BM25 + 0.25 × BGE`, a hard conflict veto, and average-linkage clustering.
2. Resolve event pairs with independent equivalence-support and forbidden-risk channels plus a non-compensatory hard gate.
3. Resolve relation pairs in risk order: Conflict → Equivalent → Complementary/Unrelated.
4. Reconstruct case graphs with constrained disjoint sets and complete provenance.
5. Build a structural backbone from BGE candidates and direction-aware evidence.
6. Recover admissible ROOT_CAUSE residual associations and apply one maximum-weight matching. Residual topics are never recycled as candidates.
7. Assemble immutable case facts, canonical concepts, diagnostic patterns, and hierarchy topics into one graph view.

See [ALGORITHM_ALIGNMENT.md](ALGORITHM_ALIGNMENT.md) for the paper-to-code mapping and the consistency fixes applied during repository cleanup.

## Repository layout

```text
configs/                    Example method configuration
schemas/                    Input JSON schemas
scripts/                    Four data-independent command-line entry points
src/casefusion/             Case resolution, state fusion, and reconstruction
src/hierarchy_fusion/       Canonicalization, patterns, hierarchy, and assembly
tests/                      Synthetic invariant and formula tests
```

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

To generate BGE embeddings locally:

```bash
python -m pip install -e ".[transformer]"
```

## Inputs

The repository does not ship manuscript data or finished MCPG artifacts because they contain
industry-confidential information. To generate MCPGs from source data, use the code provided in
the preceding MCPG repository:
[event_causality_identification](https://github.com/TristanRui/event_causality_identification).

After generating or preparing the required inputs locally, supply the following files:

- record JSONL: one object per line with `document_id` and `text`;
- local MCPG JSONL: `document_id`, `text`, `nodes`, and `relations` following `schemas/document_graph.schema.json`;
- event decisions JSONL: source event pairs with `equivalent`, `related`, or `forbidden` state;
- relation decisions JSONL: source relation pairs with `equivalent`, `complementary`, `conflicting`, or `unrelated` state;
- BGE NPZ cache: aligned `document_ids` and normalized `embeddings` arrays.

The `.gitignore` excludes data, models, caches, artifacts, and outputs by default.

## Running the mainline

Create document embeddings:

```bash
python scripts/encode_records.py \
  --records data/records.jsonl \
  --output artifacts/embeddings/document_bge.npz
```

Resolve record identities:

```bash
python scripts/resolve_cases.py \
  --records data/records.jsonl \
  --embeddings artifacts/embeddings/document_bge.npz \
  --output artifacts/case_assignments.json
```

Reconstruct case graphs from frozen state decisions:

```bash
python scripts/reconstruct_case_graphs.py \
  --local-graphs data/local_mcpg.jsonl \
  --case-assignments artifacts/case_assignments.json \
  --event-decisions artifacts/event_decisions.jsonl \
  --relation-decisions artifacts/relation_decisions.jsonl \
  --output-dir artifacts/case_graphs
```

Copy `configs/config.example.yaml`, update its paths, and build the hierarchy:

```bash
python scripts/build_hierarchy.py \
  --config configs/config.local.yaml \
  --output-dir artifacts/hierarchy
```

## Python APIs

- `casefusion.resolve_cases`: frozen case-identity scoring and clustering.
- `casefusion.fusion.EventStateFusionModel`: two-channel ExtraTrees event-state model.
- `casefusion.fusion.ComplementaryRelationClassifier`: final residual relation classifier.
- `casefusion.fusion.decide_relation_state`: risk-prioritized relation decision rule.
- `casefusion.reconstruction.build_case_graphs`: constraint-preserving reconstruction.
- `hierarchy_fusion.coarsening.build_route_b_hierarchy`: final structure-first hierarchy.

## Verification

```bash
ruff check src scripts tests
python -m pytest
```

The synthetic tests assert the frozen case weights, non-compensatory event and relation gates, provenance preservation, and the degree-one non-transitive residual constraint.

## Data and licensing notes

No manuscript dataset or trained model weights are included. BGE model files are obtained
separately from their upstream provider.

Copyright (C) 2026 Wang Ruikang. The code and documentation in this repository are licensed
under the [GNU General Public License v3.0 only](LICENSE) (`GPL-3.0-only`). Distributed copies
and modified versions must preserve the applicable GPL notices and source-code obligations.
Datasets, finished MCPG artifacts, model weights, and confidential materials excluded from this
repository are not part of this licensed release.
