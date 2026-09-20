from __future__ import annotations

import re

ASCII_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-_/][A-Za-z0-9]+)*|\d+(?:\.\d+)?")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def domain_tokens(text: str) -> list[str]:
    """Tokenize Chinese maintenance text without an external segmenter.

    Character unigrams retain component numbers and short technical terms;
    character bigrams improve phrase specificity. ASCII part numbers and
    decimal values are preserved as complete tokens.
    """

    normalized = text.casefold()
    tokens = ASCII_TOKEN_RE.findall(normalized)
    for sequence in CHINESE_RE.findall(normalized):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def character_dice_similarity(left: str, right: str, ngram_size: int = 2) -> float:
    """Return Sørensen-Dice similarity over whitespace-free character n-grams."""

    def ngrams(value: str) -> set[str]:
        normalized = "".join(value.casefold().split())
        if not normalized:
            return set()
        if len(normalized) < ngram_size:
            return {normalized}
        return {
            normalized[index : index + ngram_size]
            for index in range(len(normalized) - ngram_size + 1)
        }

    left_ngrams = ngrams(left)
    right_ngrams = ngrams(right)
    if not left_ngrams or not right_ngrams:
        return 0.0
    return 2.0 * len(left_ngrams & right_ngrams) / (len(left_ngrams) + len(right_ngrams))
