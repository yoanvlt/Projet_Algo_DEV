"""Text normalization helpers shared by BM25 and the fallback embedder."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "between",
    "by",
    "can",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "over",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "with",
}


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Return lowercase alphanumeric tokens.

    The project intentionally keeps tokenization simple so the BM25 baseline is
    explainable during the oral presentation.
    """

    tokens = TOKEN_RE.findall(text.lower())
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]
