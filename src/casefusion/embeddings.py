"""Frozen BGE text embeddings used by case resolution and hierarchy building."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def encode_bge_texts(
    texts: Sequence[str],
    *,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 16,
    maximum_length: int = 512,
    device: str = "auto",
    local_files_only: bool = False,
) -> np.ndarray:
    """Encode texts with normalized CLS vectors from the manuscript BGE model."""

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("BGE encoding requires the optional 'transformer' dependencies") from exc
    resolved_device = (
        "cuda:0"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(
        resolved_device
    )
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=maximum_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            vectors = model(**encoded).last_hidden_state[:, 0]
            vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1)
            batches.append(vectors.cpu().numpy().astype(np.float32))
    if not batches:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0)
