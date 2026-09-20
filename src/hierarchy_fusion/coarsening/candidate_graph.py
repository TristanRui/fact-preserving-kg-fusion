from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_ann_candidate_pairs(
    case_ids: list[str],
    vectors: dict[str, np.ndarray],
    *,
    top_k: int = 12,
) -> set[tuple[str, str]]:
    if len(case_ids) < 2:
        return set()
    matrix = np.stack([vectors[case_id] for case_id in case_ids])
    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(top_k + 1, len(case_ids)),
    )
    _, indices = model.fit(matrix).kneighbors(matrix)
    pairs: set[tuple[str, str]] = set()
    for left_idx, row in enumerate(indices):
        for right_idx_raw in row:
            right_idx = int(right_idx_raw)
            if left_idx == right_idx:
                continue
            pairs.add(tuple(sorted((case_ids[left_idx], case_ids[right_idx]))))
    return pairs
