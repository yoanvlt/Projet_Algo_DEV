"""Evaluation metrics and comparison helpers.

All metrics are implemented from scratch so they stay explainable for the oral:

- ``precision@K`` / ``recall@K`` use *binary* relevance (grade 3, see
  :data:`vector_search_memoire.dataset.BINARY_RELEVANCE_THRESHOLD`);
- ``MAP`` (mean average precision) also uses binary relevance, but rewards putting
  relevant documents *early* in the ranking;
- ``nDCG@K`` uses the *graded* relevance (0-3) so a near-miss (grade 1/2) still
  earns partial credit, which is what makes the metric discriminate between
  engines that all reach the same precision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .bm25 import BM25Index
from .dataset import Document, Query
from .embeddings import BaseEmbedder
from .vector_index import ExactVectorIndex

# How deep each engine ranking is scored. precision@K / recall@K read the first K,
# but MAP and recall need to see relevant documents that fall past K, so the engines
# return a deeper list for metric computation.
EVAL_DEPTH = 100


@dataclass(frozen=True)
class EngineMetrics:
    precision_at_k: float
    recall_at_k: float
    relevant_found: int
    relevant_total: int


@dataclass(frozen=True)
class RankingMetrics:
    """Full metric bundle for one ranked list of document indices."""

    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    average_precision: float
    relevant_found: int
    relevant_total: int


def relevant_indices(documents: list[Document], query: Query) -> set[int]:
    return {idx for idx, document in enumerate(documents) if query.is_relevant(document)}


def precision_recall_at_k(results: list[int], relevant: set[int], k: int) -> EngineMetrics:
    top_k = results[:k]
    found = sum(1 for idx in top_k if idx in relevant)
    precision = found / k if k else 0.0
    recall = found / len(relevant) if relevant else 0.0
    return EngineMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        relevant_found=found,
        relevant_total=len(relevant),
    )


def _dcg(gains: list[float]) -> float:
    """Discounted cumulative gain with the standard ``log2(rank + 1)`` discount."""

    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked_grades: list[int], ideal_grades: list[int], k: int) -> float:
    """nDCG@K from graded relevance, using ``gain = 2**grade - 1``."""

    gains = [2 ** grade - 1 for grade in ranked_grades[:k]]
    ideal = [2 ** grade - 1 for grade in sorted(ideal_grades, reverse=True)[:k]]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def average_precision(ranked_indices: list[int], relevant: set[int]) -> float:
    """Average precision over binary relevance for one query (basis of MAP)."""

    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def evaluate_ranking(
    ranked_indices: list[int],
    documents: list[Document],
    query: Query,
    k: int = 10,
    ndcg_k: int = 10,
) -> RankingMetrics:
    """Compute every metric for one engine's ranking. Engine-agnostic and reused
    for BM25, dense and hybrid so the three regimes are strictly comparable."""

    relevant = relevant_indices(documents, query)
    top_k = ranked_indices[:k]
    found = sum(1 for idx in top_k if idx in relevant)
    precision = found / k if k else 0.0
    recall = found / len(relevant) if relevant else 0.0
    ranked_grades = [query.relevance_grade(documents[idx]) for idx in ranked_indices]
    ideal_grades = [query.relevance_grade(document) for document in documents]
    return RankingMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        ndcg_at_k=ndcg_at_k(ranked_grades, ideal_grades, ndcg_k),
        average_precision=average_precision(ranked_indices, relevant),
        relevant_found=found,
        relevant_total=len(relevant),
    )


def evaluate_query(
    documents: list[Document],
    query: Query,
    bm25: BM25Index,
    exact_index: ExactVectorIndex,
    embedder: BaseEmbedder,
    k: int = 10,
    depth: int | None = None,
) -> dict[str, object]:
    # ``depth`` is the ranking length scored for MAP/recall. It is set to the hybrid
    # candidate_count by the experiment runner so BM25/dense MAP is comparable to the
    # hybrid MAP (which is structurally capped at candidate_count); see evaluate_regimes.
    depth = min(EVAL_DEPTH if depth is None else depth, len(documents))

    bm25_start = time.perf_counter()
    bm25_results = bm25.search(query.text, k=depth)
    bm25_seconds = time.perf_counter() - bm25_start

    query_vector = embedder.encode([query.text])[0]
    vector_start = time.perf_counter()
    vector_results = exact_index.search(query_vector, k=depth)
    vector_seconds = time.perf_counter() - vector_start

    bm25_indices = [result.index for result in bm25_results]
    vector_indices = [result.index for result in vector_results]
    bm25_metrics = evaluate_ranking(bm25_indices, documents, query, k=k)
    vector_metrics = evaluate_ranking(vector_indices, documents, query, k=k)

    return {
        "query_id": query.query_id,
        "query": query.text,
        "topic": query.topic or "all",
        "style": query.style,
        "relevant_concepts": ", ".join(query.relevant_concepts),
        "relevant_total": bm25_metrics.relevant_total,
        "bm25_precision_at_k": bm25_metrics.precision_at_k,
        "bm25_recall_at_k": bm25_metrics.recall_at_k,
        "bm25_ndcg_at_k": bm25_metrics.ndcg_at_k,
        "bm25_average_precision": bm25_metrics.average_precision,
        "bm25_seconds": bm25_seconds,
        "vector_precision_at_k": vector_metrics.precision_at_k,
        "vector_recall_at_k": vector_metrics.recall_at_k,
        "vector_ndcg_at_k": vector_metrics.ndcg_at_k,
        "vector_average_precision": vector_metrics.average_precision,
        "vector_seconds": vector_seconds,
        "bm25_top_docs": ",".join(documents[idx].doc_id for idx in bm25_indices[:k]),
        "vector_top_docs": ",".join(documents[idx].doc_id for idx in vector_indices[:k]),
        "note": query.note,
    }


def evaluate_all(
    documents: list[Document],
    queries: list[Query],
    bm25: BM25Index,
    exact_index: ExactVectorIndex,
    embedder: BaseEmbedder,
    k: int = 10,
    depth: int | None = None,
) -> list[dict[str, object]]:
    return [
        evaluate_query(documents, query, bm25, exact_index, embedder, k=k, depth=depth)
        for query in queries
    ]


def mean_metric(rows: list[dict[str, object]], metric: str) -> float:
    values = [float(row[metric]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def relevance_summary(
    documents: list[Document], queries: list[Query]
) -> dict[str, float]:
    """Min/median/max of the number of binary-relevant documents per query.

    Logged after the relevance tightening so we can show precision@K no longer
    saturates: a small, dispersed gold set is what makes the metrics meaningful.
    """

    counts = [len(relevant_indices(documents, query)) for query in queries]
    counts_array = np.array(counts, dtype=float)
    return {
        "n_queries": len(queries),
        "relevant_per_query_min": float(counts_array.min()) if counts else 0.0,
        "relevant_per_query_median": float(np.median(counts_array)) if counts else 0.0,
        "relevant_per_query_max": float(counts_array.max()) if counts else 0.0,
        "relevant_per_query_mean": float(counts_array.mean()) if counts else 0.0,
    }


def compare_ann_recall(
    queries: list[Query],
    embedder: BaseEmbedder,
    exact_index: ExactVectorIndex,
    ann_index,
    k: int = 5,
) -> dict[str, float]:
    recalls: list[float] = []
    exact_seconds: list[float] = []
    ann_seconds: list[float] = []

    for query in queries:
        query_vector = embedder.encode([query.text])[0]

        exact_start = time.perf_counter()
        exact_results = exact_index.search(query_vector, k=k)
        exact_seconds.append(time.perf_counter() - exact_start)

        ann_start = time.perf_counter()
        ann_results = ann_index.search(query_vector, k=k)
        ann_seconds.append(time.perf_counter() - ann_start)

        exact_set = {result.index for result in exact_results}
        ann_set = {result.index for result in ann_results}
        recalls.append(len(exact_set & ann_set) / k if k else 0.0)

    return {
        "ann_recall_at_k_vs_exact": float(np.mean(recalls)) if recalls else 0.0,
        "exact_mean_seconds": float(np.mean(exact_seconds)) if exact_seconds else 0.0,
        "ann_mean_seconds": float(np.mean(ann_seconds)) if ann_seconds else 0.0,
    }


def dimensionality_concentration(
    dimensions: list[int] | None = None,
    n_points: int = 1_000,
    random_state: int = 42,
) -> list[dict[str, float]]:
    """Measure how random pairwise distances concentrate as dimension grows."""

    if dimensions is None:
        dimensions = [2, 10, 50, 200, 768]

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float]] = []
    for dim in dimensions:
        points = rng.normal(size=(n_points, dim)).astype(np.float32)
        query = rng.normal(size=(1, dim)).astype(np.float32)
        distances = np.linalg.norm(points - query, axis=1)
        rows.append(
            {
                "dimension": float(dim),
                "min_distance": float(np.min(distances)),
                "mean_distance": float(np.mean(distances)),
                "max_distance": float(np.max(distances)),
                "std_distance": float(np.std(distances)),
                "relative_spread": float((np.max(distances) - np.min(distances)) / np.mean(distances)),
                "coefficient_of_variation": float(np.std(distances) / np.mean(distances)),
            }
        )
    return rows
