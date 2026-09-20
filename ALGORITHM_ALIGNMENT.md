# Paper-to-code alignment

This audit uses the supplied 40-page manuscript as the specification. Page numbers below refer to the PDF pages.

| Manuscript method | Implementation | Locked behavior |
|---|---|---|
| §3.3, pp. 10–11 | `casefusion.case_resolution` | symmetric BM25 (`k1=1.5`, `b=0.75`), BGE cosine, weights `0.75/0.25`, threshold `0.40817840781035897`, hard conflict veto, average linkage |
| §3.4, pp. 11–13 | `casefusion.fusion.event_states` | semantic/role/state/scope features, independent equivalence and forbidden channels, hard conflict gate, ordered three-state decision |
| §3.5, pp. 14–16 | `casefusion.fusion.relation_states` | direct and reverse endpoints, weaker-endpoint constraint, conflict-first decision, non-compensatory equivalence, complementary/unrelated residual classifier |
| §3.6, pp. 16–19 | `casefusion.reconstruction.case_graphs` | component-wide blocked-pair checks, endpoint projection, state-to-graph operations, source provenance |
| §4.2, pp. 20–22 | `hierarchy_fusion.semantics` and `patterns` | role-constrained canonicalization, numeric safety, normalized diagnostic direction, paths up to four edges |
| §4.3, pp. 22–24 | `hierarchy_fusion.coarsening` and `fusion` | top-12 BGE candidates, `0.22/0.78` semantic/evidence score, threshold `0.19694262623786926`, bounded k-core backbone, ROOT_CAUSE TF-IDF residual |
| §4.4, pp. 24–26 | `route_b_hierarchy` | strongest leaf witness per root pair, maximum-weight matching, degree ≤ 1, no recursive residual topics |
| §4.5, p. 26 | `assembly.unified_graph` | case facts remain immutable; concepts, patterns, and topics add reference or hierarchy links only |

## Corrections made during cleanup

1. The former public relation decision function used an older compensatory score and inferred conflict from reason flags. It now uses the final non-compensatory endpoint/semantic score, a separate conflict score, and the paper's Conflict → Equivalent → Complementary/Unrelated order.
2. Structural-backbone construction and residual risk typing previously lived under `baselines/` and `evaluation/`. They are part of the proposed method, so they now live in `coarsening/structural_backbone.py` and `coarsening/risk.py`.
3. The recursive residual implementation was an ablation, not the proposed algorithm. It has been removed; the remaining implementation creates exactly one residual matching layer.
4. Case-resolution parameters previously depended on generated benchmark files. They are now explicit immutable defaults in `CaseResolutionConfig`.
5. Hierarchy configuration previously referenced embedding caches under ablation-result directories. The cleaned pipeline accepts user-supplied embedding paths without depending on experiments or results.

## Terminology ledger

| Canonical term | Code name |
|---|---|
| maintenance record | `document_id` |
| case identity | case assignment |
| local MCPG | local graph input |
| reconstructed case graph | `CaseGraph` |
| event state | `equivalent`, `related`, `forbidden` |
| relation state | `equivalent`, `complementary`, `conflicting`, `unrelated` |
| canonical diagnostic concept | concept |
| diagnostic pattern | `PatternInstance` |
| structural backbone | backbone root set |
| residual association | ROOT_CAUSE sparse witness |
| parent topic | hierarchy-only `DiagnosticUnit` |

The repository deliberately avoids calling a parent topic a case or treating a cross-case association as a factual relation.
