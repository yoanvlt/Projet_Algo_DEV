"""Two-stage hybrid retrieval: BM25 recall, then re-ranking.

The course states twice that "BM25 (filtering) + embeddings (re-ranking) beats
either approach alone" but never builds or measures it. This module builds it:

1. **Stage 1 - recall.** BM25 (from scratch) returns the top-N candidates. It is
   cheap and good at catching exact-vocabulary / domain matches.
2. **Stage 2 - re-ranking.** A re-ranker re-orders only those N candidates into the
   final top-K. Two variants are provided:

   - ``bi-encoder`` : cosine similarity between the query embedding and the *already
     computed* document embeddings. Almost free (one matrix-vector product over N
     candidates).
   - ``cross-encoder`` : a real ``sentence-transformers`` cross-encoder that scores
     each ``(query, candidate)`` pair jointly. This is the "production" option. A
     deterministic, clearly-labelled fallback is used when the model cannot be
     loaded so the demo still runs offline.

Because stage 2 only ever re-orders stage-1 candidates, the hybrid top-K is always
a permutation of (a prefix of) the BM25 candidate set - an invariant the tests
check.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np

from .bm25 import BM25Index
from .embeddings import BaseEmbedder
from .text import tokenize


@dataclass(frozen=True)
class HybridResult:
    index: int
    final_rank: int
    stage1_rank: int
    rerank_score: float
    bm25_score: float

    @property
    def rank_delta(self) -> int:
        """How many positions re-ranking moved the document.

        Positive = promoted (moved up), negative = demoted. Used by the Streamlit
        demo to highlight what re-ranking changed.
        """

        return self.stage1_rank - self.final_rank


def _min_max(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    low = float(scores.min())
    high = float(scores.max())
    if high - low < 1e-12:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


class BaseReranker:
    name = "base"
    is_fallback = False

    def metadata(self) -> dict[str, object]:
        """Reproducibility metadata recorded in the run summaries.

        ``is_fallback`` is always reported so a fallback run can never be mistaken
        for a real cross-encoder run.
        """

        return {"name": self.name, "is_fallback": self.is_fallback, "kind": type(self).__name__}

    def score(
        self,
        query_text: str,
        query_vector: np.ndarray,
        candidate_indices: list[int],
        doc_texts: list[str],
        doc_vectors: np.ndarray,
        bm25_scores: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class BiEncoderReranker(BaseReranker):
    """Re-rank candidates by cosine similarity of precomputed embeddings."""

    name = "bi-encoder"

    def score(self, query_text, query_vector, candidate_indices, doc_texts, doc_vectors, bm25_scores):
        candidate_vectors = doc_vectors[np.asarray(candidate_indices, dtype=np.int64)]
        return candidate_vectors @ query_vector.astype(np.float32)


class CrossEncoderReranker(BaseReranker):
    """Real sentence-transformers cross-encoder scoring each (query, doc) pair."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        import sentence_transformers
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model_name = model_name
        self.library_version = getattr(sentence_transformers, "__version__", "unknown")
        self.model = CrossEncoder(model_name)
        # Best-effort HF revision (commit hash); not always available offline.
        self.revision = getattr(getattr(self.model, "config", None), "_commit_hash", None)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "is_fallback": self.is_fallback,
            "kind": type(self).__name__,
            "model_name": self.model_name,
            "sentence_transformers_version": self.library_version,
            "revision": self.revision,
        }

    def score(self, query_text, query_vector, candidate_indices, doc_texts, doc_vectors, bm25_scores):
        pairs = [(query_text, doc_texts[idx]) for idx in candidate_indices]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)


class CrossEncoderFallbackReranker(BaseReranker):
    """Deterministic offline stand-in for the cross-encoder (clearly a plan B).

    It blends the bi-encoder cosine with a lexical token-overlap term, so it scores
    each (query, candidate) pair without any heavy model while still combining
    semantic and lexical evidence. It is *not* a real cross-encoder; results
    obtained with it are labelled as a fallback, never as the headline numbers.
    """

    name = "cross-encoder-fallback"
    is_fallback = True

    def __init__(self, cosine_weight: float = 0.6, lexical_weight: float = 0.4) -> None:
        self.cosine_weight = cosine_weight
        self.lexical_weight = lexical_weight

    def score(self, query_text, query_vector, candidate_indices, doc_texts, doc_vectors, bm25_scores):
        candidate_vectors = doc_vectors[np.asarray(candidate_indices, dtype=np.int64)]
        cosine = candidate_vectors @ query_vector.astype(np.float32)
        query_tokens = set(tokenize(query_text))
        overlap = np.array(
            [
                (len(query_tokens & set(tokenize(doc_texts[idx]))) / len(query_tokens))
                if query_tokens
                else 0.0
                for idx in candidate_indices
            ],
            dtype=np.float32,
        )
        return self.cosine_weight * _min_max(cosine) + self.lexical_weight * overlap


def cross_encoder_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def build_reranker(
    kind: str = "auto",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> BaseReranker:
    """Create a re-ranker.

    kind values:
    - ``bi-encoder`` : cosine re-ranking on precomputed embeddings;
    - ``cross-encoder`` : real cross-encoder, deterministic fallback if it fails;
    - ``auto`` : cross-encoder if available, otherwise the fallback.
    """

    if kind == "bi-encoder":
        return BiEncoderReranker()
    if kind in ("cross-encoder", "auto"):
        if cross_encoder_available():
            try:
                return CrossEncoderReranker(model_name=model_name)
            except Exception:
                return CrossEncoderFallbackReranker()
        return CrossEncoderFallbackReranker()
    raise ValueError(f"Unknown reranker kind: {kind}")


class HybridSearcher:
    """BM25 recall (top-N) followed by re-ranking into the final top-K."""

    def __init__(
        self,
        bm25: BM25Index,
        doc_vectors: np.ndarray,
        doc_texts: list[str],
        embedder: BaseEmbedder,
        reranker: BaseReranker,
        candidate_count: int = 50,
    ) -> None:
        self.bm25 = bm25
        self.doc_vectors = doc_vectors.astype(np.float32)
        self.doc_texts = doc_texts
        self.embedder = embedder
        self.reranker = reranker
        self.candidate_count = candidate_count

    @property
    def name(self) -> str:
        return f"hybrid[bm25->{self.reranker.name}]"

    def search(self, query_text: str, k: int = 10) -> list[HybridResult]:
        candidates = self.bm25.search(query_text, k=self.candidate_count)
        if not candidates:
            return []
        candidate_indices = [result.index for result in candidates]
        bm25_scores = np.array([result.score for result in candidates], dtype=np.float32)
        query_vector = self.embedder.encode([query_text])[0]

        rerank_scores = self.reranker.score(
            query_text,
            query_vector,
            candidate_indices,
            self.doc_texts,
            self.doc_vectors,
            bm25_scores,
        )

        order = np.argsort(-rerank_scores, kind="stable")
        results: list[HybridResult] = []
        for final_rank, pos in enumerate(order[:k], start=1):
            results.append(
                HybridResult(
                    index=candidate_indices[pos],
                    final_rank=final_rank,
                    stage1_rank=int(pos) + 1,
                    rerank_score=float(rerank_scores[pos]),
                    bm25_score=float(bm25_scores[pos]),
                )
            )
        return results
