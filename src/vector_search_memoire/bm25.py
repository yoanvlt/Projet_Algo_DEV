"""Small BM25 implementation written from scratch for explainability.

It uses a proper **inverted index** (posting lists): for each term we store the
list of ``(document index, term frequency)`` pairs. Scoring a query then only
visits the documents that actually contain the query terms, instead of scanning
the whole corpus for every term. This is the textbook BM25 data structure and it
makes the BM25 recall stage cheap even when the corpus is large.

The scoring formula and the produced rankings are *identical* to the naive
full-scan implementation: for a given document the per-term contributions are
accumulated in the same (query-term) order, so the floating-point results match
exactly. ``tests/test_extensions.py`` checks this against a reference scorer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from .text import tokenize


@dataclass(frozen=True)
class SearchResult:
    index: int
    score: float


class BM25Index:
    """Okapi BM25 over a list of documents, backed by an inverted index.

    The implementation uses the common non-negative IDF variant:
    log(1 + (N - df + 0.5) / (df + 0.5)).
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize(document) for document in documents]
        self.doc_lengths = np.array([len(tokens) for tokens in self.tokenized], dtype=np.float32)
        self.avg_doc_length = float(np.mean(self.doc_lengths)) if len(documents) else 0.0
        # Inverted index: term -> list of (doc_idx, term_frequency), doc_idx ascending.
        self.postings = self._build_postings()
        self.document_frequency = {term: len(plist) for term, plist in self.postings.items()}
        self.idf = self._idf()

    def _build_postings(self) -> dict[str, list[tuple[int, int]]]:
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_idx, tokens in enumerate(self.tokenized):
            for term, frequency in Counter(tokens).items():
                postings.setdefault(term, []).append((doc_idx, frequency))
        return postings

    def _idf(self) -> dict[str, float]:
        n_docs = len(self.documents)
        return {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in self.document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not query_terms or self.avg_doc_length == 0:
            return scores

        for term in query_terms:
            posting_list = self.postings.get(term)
            if posting_list is None:
                continue
            idf = self.idf[term]
            for doc_idx, frequency in posting_list:
                doc_length = self.doc_lengths[doc_idx]
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * doc_length / self.avg_doc_length
                )
                scores[doc_idx] += idf * frequency * (self.k1 + 1.0) / denominator
        return scores

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        scores = self.scores(query)
        if len(scores) == 0:
            return []
        k = min(k, len(scores))
        candidate_indices = np.argpartition(-scores, kth=k - 1)[:k]
        ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
        return [SearchResult(index=int(idx), score=float(scores[idx])) for idx in ordered]
