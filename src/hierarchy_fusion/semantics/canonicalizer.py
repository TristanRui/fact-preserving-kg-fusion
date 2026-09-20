from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from hierarchy_fusion.data.case_graph import CaseGraph, EventNode

_PUNCT = re.compile(r"[\s，。；、：:,.!?！？（）()【】\[\]<>《》“”\"'‘’\-—_]+")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:mm|cm|m|MPa|kPa|Pa|N·m|Nm|℃|°C|%)?", re.I)
_RULES = (
    (re.compile(r"换装|换新|替换"), "更换"),
    (re.compile(r"拧紧|压紧|重新紧固"), "紧固"),
    (re.compile(r"复测|重新测量"), "测量"),
    (re.compile(r"气密性试验|气密检查"), "气密试验"),
    (re.compile(r"清洁"), "清理"),
    (re.compile(r"拆卸|拆开"), "分解"),
    (re.compile(r"重新安装|回装"), "复装"),
)


def normalize(text: str) -> str:
    value = _PUNCT.sub("", str(text or "").strip().lower())
    for pattern, replacement in _RULES:
        value = pattern.sub(replacement, value)
    return value


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(sorted(match.group(0).lower() for match in _NUMBER.finditer(text or "")))


def _number_safe(left: str, right: str) -> bool:
    a, b = _numbers(left), _numbers(right)
    return not (a and b and a != b)


def _char_ngram_containment(left: str, right: str, size: int = 2) -> float:
    def grams(value: str) -> set[str]:
        if not value:
            return set()
        if len(value) < size:
            return {value}
        return {value[idx : idx + size] for idx in range(len(value) - size + 1)}

    a, b = grams(left), grams(right)
    return len(a & b) / max(1, min(len(a), len(b)))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


@dataclass(slots=True)
class CanonicalizationResult:
    node_to_concept: dict[str, str]
    concepts: dict[str, dict]
    review_pairs: list[dict]
    stats: dict


def _concept_id(role: str, label: str) -> str:
    return f"C_{role[:3]}_{hashlib.sha1(f'{role}|{label}'.encode()).hexdigest()[:12]}"


def canonicalize(
    graphs: list[CaseGraph],
    *,
    threshold: float = 0.92,
    top_k: int = 10,
    ngram_range: tuple[int, int] = (2, 4),
    max_features: int = 30000,
    semantic_vectors: dict[str, np.ndarray] | None = None,
    semantic_threshold: float = 0.90,
    semantic_top_k: int = 10,
) -> CanonicalizationResult:
    role_nodes: dict[str, list[EventNode]] = defaultdict(list)
    for graph in graphs:
        for node in graph.nodes:
            role_nodes[node.role].append(node)

    mapping: dict[str, str] = {}
    concepts: dict[str, dict] = {}
    review: list[dict] = []
    role_stats: dict[str, dict] = {}
    for role, nodes in sorted(role_nodes.items()):
        values = [normalize(node.text) or node.text.lower() for node in nodes]
        uf = _UnionFind(len(nodes))
        exact: dict[str, list[int]] = defaultdict(list)
        for idx, value in enumerate(values):
            exact[value].append(idx)
        for members in exact.values():
            for idx in members[1:]:
                uf.union(members[0], idx)

        accepted = 0
        if len(nodes) > 1 and len(set(values)) > 1:
            matrix = TfidfVectorizer(
                analyzer="char",
                ngram_range=ngram_range,
                min_df=1,
                max_features=max_features,
                sublinear_tf=True,
            ).fit_transform(values)
            model = NearestNeighbors(
                metric="cosine", algorithm="brute", n_neighbors=min(top_k + 1, len(nodes))
            )
            distances, indices = model.fit(matrix).kneighbors(matrix)
            seen: set[tuple[int, int]] = set()
            for left, (row_distances, row_indices) in enumerate(
                zip(distances, indices, strict=True)
            ):
                for distance, right_raw in zip(row_distances, row_indices, strict=True):
                    right = int(right_raw)
                    pair = (min(left, right), max(left, right))
                    if left == right or pair in seen:
                        continue
                    seen.add(pair)
                    similarity = float(1.0 - distance)
                    if similarity < threshold:
                        continue
                    safe = _number_safe(nodes[left].text, nodes[right].text)
                    if safe:
                        uf.union(left, right)
                        accepted += 1
                    review.append(
                        {
                            "role": role,
                            "left_uid": nodes[left].uid,
                            "right_uid": nodes[right].uid,
                            "left_text": nodes[left].text,
                            "right_text": nodes[right].text,
                            "similarity": round(similarity, 6),
                            "number_safe": safe,
                            "accepted": safe,
                            "evidence_type": "char_tfidf",
                        }
                    )

        semantic_accepted = 0
        if semantic_vectors and len(nodes) > 1:
            matrix = np.stack([semantic_vectors[node.uid] for node in nodes])
            model = NearestNeighbors(
                metric="cosine",
                algorithm="brute",
                n_neighbors=min(semantic_top_k + 1, len(nodes)),
            )
            distances, indices = model.fit(matrix).kneighbors(matrix)
            seen_semantic: set[tuple[int, int]] = set()
            for left, (row_distances, row_indices) in enumerate(
                zip(distances, indices, strict=True)
            ):
                for distance, right_raw in zip(row_distances, row_indices, strict=True):
                    right = int(right_raw)
                    pair = (min(left, right), max(left, right))
                    if left == right or pair in seen_semantic:
                        continue
                    seen_semantic.add(pair)
                    similarity = float(1.0 - distance)
                    if similarity < semantic_threshold:
                        continue
                    safe = _number_safe(nodes[left].text, nodes[right].text)
                    containment = _char_ngram_containment(values[left], values[right])
                    left_documents = set(
                        nodes[left].source_document_ids or (nodes[left].document_id,)
                    )
                    right_documents = set(
                        nodes[right].source_document_ids or (nodes[right].document_id,)
                    )
                    same_document = bool(left_documents & right_documents)
                    # Context BGE is candidate evidence, never the sole merge
                    # decision. Same-clause events can have cosine 1.0 while
                    # expressing different or opposite actions.
                    lexical_anchor = (
                        values[left] == values[right]
                        or values[left] in values[right]
                        or values[right] in values[left]
                        or containment >= 0.70
                    )
                    accepted_pair = (
                        safe
                        and lexical_anchor
                        and (not same_document or values[left] == values[right])
                    )
                    if accepted_pair:
                        uf.union(left, right)
                        semantic_accepted += 1
                    review.append(
                        {
                            "role": role,
                            "left_uid": nodes[left].uid,
                            "right_uid": nodes[right].uid,
                            "left_text": nodes[left].text,
                            "right_text": nodes[right].text,
                            "similarity": round(similarity, 6),
                            "number_safe": safe,
                            "char_bigram_containment": round(containment, 6),
                            "same_document": same_document,
                            "lexical_anchor": lexical_anchor,
                            "accepted": accepted_pair,
                            "evidence_type": "context_bge",
                        }
                    )

        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(nodes)):
            groups[uf.find(idx)].append(idx)
        for indexes in groups.values():
            members = [nodes[idx] for idx in indexes]
            counts = Counter(node.text.strip() for node in members)
            label = min(counts, key=lambda value: (-counts[value], -len(value), value))
            normalized_label = normalize(label) or label.lower()
            concept_id = _concept_id(role, normalized_label)
            if concept_id not in concepts:
                concepts[concept_id] = {
                    "concept_id": concept_id,
                    "role": role,
                    "label": label,
                    "normalized_label": normalized_label,
                    "member_count": 0,
                    "case_ids": set(),
                    "variants": set(),
                }
            item = concepts[concept_id]
            item["member_count"] += len(members)
            item["case_ids"].update(node.case_id for node in members)
            item["variants"].update(node.text for node in members)
            for node in members:
                mapping[node.uid] = concept_id
        role_stats[role] = {
            "node_count": len(nodes),
            "concept_count": len({mapping[node.uid] for node in nodes}),
            "semantic_pair_merge_count": accepted,
            "context_bge_pair_merge_count": semantic_accepted,
        }

    for item in concepts.values():
        item["case_count"] = len(item.pop("case_ids"))
        item["variants"] = sorted(item["variants"])
    review.sort(key=lambda row: (-row["similarity"], row["role"], row["left_uid"]))
    return CanonicalizationResult(
        node_to_concept=mapping,
        concepts=concepts,
        review_pairs=review,
        stats={
            "node_count": len(mapping),
            "concept_count": len(concepts),
            "compression_ratio": 1.0 - len(concepts) / max(1, len(mapping)),
            "roles": role_stats,
        },
    )
