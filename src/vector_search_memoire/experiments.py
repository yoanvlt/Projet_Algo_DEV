"""End-to-end experiment runner used by scripts and the Streamlit app."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np

from .bm25 import BM25Index
from .dataset import Document, Query, default_queries, generate_corpus
from .embeddings import BaseEmbedder, build_embedder
from .evaluation import (
    EVAL_DEPTH,
    compare_ann_recall,
    dimensionality_concentration,
    evaluate_all,
    evaluate_ranking,
    mean_metric,
    relevance_summary,
)
from .hybrid import HybridSearcher, build_reranker
from .vector_index import ExactVectorIndex, build_ann_index


def build_artifacts(
    n_docs: int = 2_000,
    random_state: int = 42,
    embedding_provider: str = "auto",
) -> dict[str, object]:
    documents = generate_corpus(n_docs=n_docs, random_state=random_state)
    queries = default_queries()
    texts = [document.text for document in documents]

    bm25 = BM25Index(texts)
    embedder = build_embedder(provider=embedding_provider, random_state=random_state)

    embedding_start = time.perf_counter()
    vectors = embedder.encode(texts)
    embedding_seconds = time.perf_counter() - embedding_start

    exact_index = ExactVectorIndex(vectors)
    ann_index, ann_build_seconds = build_ann_index(vectors)

    return {
        "documents": documents,
        "queries": queries,
        "texts": texts,
        "bm25": bm25,
        "embedder": embedder,
        "vectors": vectors,
        "exact_index": exact_index,
        "ann_index": ann_index,
        "embedding_seconds": embedding_seconds,
        "ann_build_seconds": ann_build_seconds,
    }


def build_hybrid_searcher(
    artifacts: dict[str, object],
    reranker_kind: str = "auto",
    candidate_count: int = 50,
) -> HybridSearcher:
    """Build a two-stage hybrid searcher on top of pre-built artifacts."""

    reranker = build_reranker(reranker_kind)
    return HybridSearcher(
        bm25=artifacts["bm25"],  # type: ignore[arg-type]
        doc_vectors=artifacts["vectors"],  # type: ignore[arg-type]
        doc_texts=artifacts["texts"],  # type: ignore[arg-type]
        embedder=artifacts["embedder"],  # type: ignore[arg-type]
        reranker=reranker,
        candidate_count=candidate_count,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Three-regime comparison: BM25 only / dense only / hybrid (bi & cross rerank)
# --------------------------------------------------------------------------- #

_METRIC_KEYS = ("precision_at_k", "recall_at_k", "ndcg_at_k", "average_precision")


def _regime_rankings(
    query: Query,
    bm25: BM25Index,
    exact_index: ExactVectorIndex,
    embedder: BaseEmbedder,
    hybrid_searchers: dict[str, HybridSearcher],
    depth: int,
) -> dict[str, list[int]]:
    # All regimes are scored at the SAME depth so MAP is comparable. ``depth`` is the
    # hybrid candidate_count, i.e. the hybrid's structural ceiling; BM25/dense are
    # truncated to the same N rather than keeping 100 results the hybrid cannot reach.
    query_vector = embedder.encode([query.text])[0]
    rankings: dict[str, list[int]] = {
        "bm25": [result.index for result in bm25.search(query.text, k=depth)],
        "dense": [result.index for result in exact_index.search(query_vector, k=depth)],
    }
    for label, searcher in hybrid_searchers.items():
        rankings[label] = [result.index for result in searcher.search(query.text, k=depth)]
    return rankings


def evaluate_regimes(
    documents: list[Document],
    queries: list[Query],
    bm25: BM25Index,
    exact_index: ExactVectorIndex,
    embedder: BaseEmbedder,
    hybrid_searchers: dict[str, HybridSearcher],
    k: int = 10,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate every regime on the same queries and the same metrics.

    Returns ``(long_rows, summary)``:

    - ``long_rows`` : one row per (query, regime) for the comparison CSV;
    - ``summary`` : mean metrics per regime, a per-style breakdown, an explicit
      conclusion on whether hybrid beats BM25 and dense, and the list of queries
      where re-ranking actually changed the BM25 ordering.
    """

    # MAP depth N = the hybrid candidate_count: the deepest the hybrid can rank. Every
    # regime is scored at this same N (= "MAP@N"), so the comparison is apples-to-apples.
    candidate_counts = [s.candidate_count for s in hybrid_searchers.values()]
    map_depth = min(candidate_counts) if candidate_counts else EVAL_DEPTH
    depth = min(map_depth, len(documents))
    regimes = ["bm25", "dense", *hybrid_searchers.keys()]

    long_rows: list[dict[str, object]] = []
    rerank_changes: list[dict[str, object]] = []
    per_regime_metrics: dict[str, dict[str, list[float]]] = {
        regime: {key: [] for key in _METRIC_KEYS} for regime in regimes
    }

    for query in queries:
        rankings = _regime_rankings(query, bm25, exact_index, embedder, hybrid_searchers, depth)
        metrics_by_regime = {
            regime: evaluate_ranking(rankings[regime], documents, query, k=k)
            for regime in regimes
        }
        for regime in regimes:
            metrics = metrics_by_regime[regime]
            for key in _METRIC_KEYS:
                per_regime_metrics[regime][key].append(getattr(metrics, key))
            long_rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.text,
                    "style": query.style,
                    "topic": query.topic or "all",
                    "regime": regime,
                    "relevant_total": metrics.relevant_total,
                    "precision_at_k": metrics.precision_at_k,
                    "recall_at_k": metrics.recall_at_k,
                    "ndcg_at_k": metrics.ndcg_at_k,
                    "average_precision": metrics.average_precision,
                }
            )

        # Did re-ranking reorder BM25's top-k? Track it for the demo / discussion.
        bm25_top = rankings["bm25"][:k]
        for regime in hybrid_searchers:
            hybrid_top = rankings[regime][:k]
            if hybrid_top != bm25_top:
                rerank_changes.append(
                    {
                        "query_id": query.query_id,
                        "style": query.style,
                        "regime": regime,
                        "ndcg_delta_vs_bm25": float(
                            metrics_by_regime[regime].ndcg_at_k
                            - metrics_by_regime["bm25"].ndcg_at_k
                        ),
                        "ap_delta_vs_bm25": float(
                            metrics_by_regime[regime].average_precision
                            - metrics_by_regime["bm25"].average_precision
                        ),
                    }
                )

    def _means(regime: str) -> dict[str, float]:
        return {
            key: float(np.mean(per_regime_metrics[regime][key])) if per_regime_metrics[regime][key] else 0.0
            for key in _METRIC_KEYS
        }

    regime_means = {regime: _means(regime) for regime in regimes}

    # Per-style breakdown of nDCG@k, so we can say *which* queries hybrid helps on.
    styles = sorted({query.style for query in queries})
    by_style: dict[str, dict[str, float]] = {}
    for style in styles:
        style_rows = [row for row in long_rows if row["style"] == style]
        by_style[style] = {
            regime: float(
                np.mean([row["ndcg_at_k"] for row in style_rows if row["regime"] == regime])
            )
            for regime in regimes
        }

    # Best hybrid variant = highest mean nDCG@k among the hybrid regimes.
    hybrid_labels = list(hybrid_searchers.keys())
    best_hybrid = max(hybrid_labels, key=lambda r: regime_means[r]["ndcg_at_k"]) if hybrid_labels else None

    conclusion: dict[str, object] = {}
    if best_hybrid is not None:
        h = regime_means[best_hybrid]["ndcg_at_k"]
        b = regime_means["bm25"]["ndcg_at_k"]
        d = regime_means["dense"]["ndcg_at_k"]
        beats_both = h > b and h > d
        # Where does the hybrid beat *both* baselines? Read per query style.
        hybrid_wins_on_styles = [
            style
            for style in styles
            if by_style[style][best_hybrid] > by_style[style]["bm25"]
            and by_style[style][best_hybrid] > by_style[style]["dense"]
        ]
        conclusion = {
            "metric": "ndcg_at_k",
            "best_hybrid_regime": best_hybrid,
            "hybrid_ndcg": h,
            "bm25_ndcg": b,
            "dense_ndcg": d,
            "hybrid_beats_bm25": h > b,
            "hybrid_beats_dense": h > d,
            "hybrid_beats_both_overall": beats_both,
            "hybrid_wins_on_styles": hybrid_wins_on_styles,
            "gain_vs_bm25": h - b,
            "gain_vs_dense": h - d,
            "verdict": (
                f"Hybrid ({best_hybrid}) nDCG@{k}={h:.3f} "
                + (
                    "beats both BM25 ({:.3f}) and dense ({:.3f}) overall"
                    if beats_both
                    else "beats BM25 ({:.3f}) but not dense ({:.3f}) overall"
                ).format(b, d)
                + (
                    f"; it does beat both on these query styles: {hybrid_wins_on_styles}."
                    if hybrid_wins_on_styles
                    else "; it never beats both on any single query style."
                )
            ),
        }

    rerankers = {label: searcher.reranker.metadata() for label, searcher in hybrid_searchers.items()}
    summary = {
        "k": k,
        "map_depth": depth,  # MAP / average_precision is computed at this depth N for all regimes
        "candidate_count": map_depth,
        "rerankers": rerankers,
        "regimes": regime_means,
        "ndcg_by_style": by_style,
        "n_rerank_changes": len(rerank_changes),
        "rerank_changes": rerank_changes,
        "conclusion": conclusion,
    }
    return long_rows, summary


def run_experiment(
    output_dir: str | Path = "outputs",
    n_docs: int = 2_000,
    random_state: int = 42,
    embedding_provider: str = "auto",
    k: int = 10,
    candidate_count: int = 50,
) -> dict[str, object]:
    output_path = Path(output_dir)
    artifacts = build_artifacts(
        n_docs=n_docs,
        random_state=random_state,
        embedding_provider=embedding_provider,
    )

    documents = artifacts["documents"]
    queries = artifacts["queries"]
    bm25 = artifacts["bm25"]
    embedder = artifacts["embedder"]
    exact_index = artifacts["exact_index"]
    ann_index = artifacts["ann_index"]

    assert isinstance(documents, list)
    assert isinstance(queries, list)
    assert isinstance(bm25, BM25Index)
    assert isinstance(embedder, BaseEmbedder)
    assert isinstance(exact_index, ExactVectorIndex)

    # MAP/recall depth = candidate_count so BM25/dense MAP matches the hybrid MAP@N.
    evaluation_rows = evaluate_all(
        documents, queries, bm25, exact_index, embedder, k=k, depth=candidate_count
    )
    dimensionality_rows = dimensionality_concentration(random_state=random_state)
    ann_summary = compare_ann_recall(queries, embedder, exact_index, ann_index, k=k)
    rel_summary = relevance_summary(documents, queries)

    # Two-stage hybrid: one bi-encoder reranker, one cross-encoder (or fallback).
    hybrid_searchers = {
        "hybrid_bi": build_hybrid_searcher(artifacts, "bi-encoder", candidate_count),
        "hybrid_cross": build_hybrid_searcher(artifacts, "auto", candidate_count),
    }
    regime_rows, regime_summary = evaluate_regimes(
        documents, queries, bm25, exact_index, embedder, hybrid_searchers, k=k
    )

    summary = {
        "n_docs": n_docs,
        "n_queries": len(queries),
        "random_state": random_state,
        "k": k,
        "candidate_count": candidate_count,
        "map_depth": candidate_count,  # *_map fields are MAP@candidate_count for all regimes
        "embedder": embedder.name,
        "ann_index": ann_index.name,
        "cross_encoder": hybrid_searchers["hybrid_cross"].reranker.name,
        "cross_encoder_is_fallback": hybrid_searchers["hybrid_cross"].reranker.is_fallback,
        "cross_encoder_metadata": hybrid_searchers["hybrid_cross"].reranker.metadata(),
        "embedding_seconds": artifacts["embedding_seconds"],
        "ann_build_seconds": artifacts["ann_build_seconds"],
        "relevance": rel_summary,
        "bm25_mean_precision_at_k": mean_metric(evaluation_rows, "bm25_precision_at_k"),
        "bm25_mean_recall_at_k": mean_metric(evaluation_rows, "bm25_recall_at_k"),
        "bm25_mean_ndcg_at_k": mean_metric(evaluation_rows, "bm25_ndcg_at_k"),
        "bm25_map": mean_metric(evaluation_rows, "bm25_average_precision"),
        "vector_mean_precision_at_k": mean_metric(evaluation_rows, "vector_precision_at_k"),
        "vector_mean_recall_at_k": mean_metric(evaluation_rows, "vector_recall_at_k"),
        "vector_mean_ndcg_at_k": mean_metric(evaluation_rows, "vector_ndcg_at_k"),
        "vector_map": mean_metric(evaluation_rows, "vector_average_precision"),
        # Compact view only; the full per-query / per-style detail lives in
        # regime_summary.json (avoid duplicating ~340 lines of JSON here).
        "regime_comparison": {
            "detail_file": "regime_summary.json",
            "regimes": regime_summary["regimes"],
            "conclusion": regime_summary["conclusion"],
        },
        **ann_summary,
    }

    write_csv(output_path / "evaluation.csv", evaluation_rows)
    write_csv(output_path / "dimensionality.csv", dimensionality_rows)
    write_csv(output_path / "regime_comparison.csv", regime_rows)
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_path / "regime_summary.json").write_text(
        json.dumps(regime_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    vectors = artifacts["vectors"]
    if isinstance(vectors, np.ndarray):
        np.save(output_path / "embeddings.npy", vectors)

    return summary
