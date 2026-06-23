"""Tests for the new work: graded relevance, nDCG/MAP, hybrid pipeline, scaling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vector_search_memoire.bm25 import BM25Index
from vector_search_memoire.dataset import (
    BINARY_RELEVANCE_THRESHOLD,
    default_queries,
    generate_corpus,
)
from vector_search_memoire.embeddings import SemanticHashingEmbedder
from vector_search_memoire.evaluation import (
    average_precision,
    evaluate_ranking,
    ndcg_at_k,
    relevance_summary,
)
from vector_search_memoire.hybrid import (
    BiEncoderReranker,
    CrossEncoderFallbackReranker,
    HybridSearcher,
    build_reranker,
    cross_encoder_available,
)
from vector_search_memoire.scale import benchmark_scale, crossover_size, make_unit_vectors


# --------------------------------------------------------------------------- #
# A. Graded relevance / metrics
# --------------------------------------------------------------------------- #
def test_relevance_sets_are_small_and_dispersed() -> None:
    """After tightening, gold sets must be small (~5-30) and not saturate."""

    documents = generate_corpus(n_docs=2_000, random_state=42)
    queries = default_queries()
    summary = relevance_summary(documents, queries)
    assert summary["relevant_per_query_min"] >= 1
    assert summary["relevant_per_query_max"] <= 40  # was 100-688 before tightening
    assert 4 <= summary["relevant_per_query_median"] <= 30


def test_relevance_grades_are_bounded_and_ordered() -> None:
    documents = generate_corpus(n_docs=500, random_state=42)
    query = default_queries()[3]  # Q04, healthcare patient_triage / hospital
    grades = [query.relevance_grade(doc) for doc in documents]
    assert set(grades) <= {0, 1, 2, 3}
    # Every binary-relevant doc reaches the threshold grade and matches all facets.
    for doc, grade in zip(documents, grades):
        if grade >= BINARY_RELEVANCE_THRESHOLD:
            assert doc.topic == query.topic
            assert doc.primary_concept == query.relevant_concepts[0]
            assert doc.domain == query.relevant_domain


def test_ndcg_matches_hand_computed_value() -> None:
    # ranked grades [2, 3, 0], ideal ordering [3, 2, 0]; gain = 2**g - 1.
    ranked = [2, 3, 0]
    ideal = [3, 2, 0]
    dcg = 3 / math.log2(2) + 7 / math.log2(3) + 0 / math.log2(4)
    idcg = 7 / math.log2(2) + 3 / math.log2(3) + 0 / math.log2(4)
    expected = dcg / idcg
    assert ndcg_at_k(ranked, ideal, k=3) == pytest.approx(expected, rel=1e-9)
    assert ndcg_at_k(ranked, ideal, k=3) == pytest.approx(0.833997, rel=1e-4)


def test_average_precision_matches_hand_computed_value() -> None:
    # relevant at ranks 2 and 4 -> AP = (1/2 + 2/4) / 3.
    ap = average_precision([5, 1, 9, 2], {1, 2, 3})
    assert ap == pytest.approx((0.5 + 0.5) / 3, rel=1e-9)


def test_evaluate_ranking_perfect_ranking_scores_one() -> None:
    documents = generate_corpus(n_docs=300, random_state=42)
    query = default_queries()[3]
    relevant = [i for i, d in enumerate(documents) if query.is_relevant(d)]
    assert relevant, "need at least one relevant doc for this test"
    # A ranking that puts all relevant docs first must reach perfect MAP and high nDCG.
    others = [i for i in range(len(documents)) if i not in set(relevant)]
    ranking = relevant + others
    metrics = evaluate_ranking(ranking, documents, query, k=len(relevant))
    assert metrics.precision_at_k == pytest.approx(1.0)
    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.average_precision == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# A bis. BM25 inverted index must match the naive full-scan reference exactly
# --------------------------------------------------------------------------- #
def _reference_bm25_scores(index: BM25Index, query: str) -> np.ndarray:
    """Naive full-scan BM25 (the pre-inverted-index reference implementation)."""

    from collections import Counter

    from vector_search_memoire.text import tokenize as _tok

    term_frequencies = [Counter(tokens) for tokens in index.tokenized]
    query_terms = _tok(query)
    scores = np.zeros(len(index.documents), dtype=np.float32)
    if not query_terms or index.avg_doc_length == 0:
        return scores
    for term in query_terms:
        if term not in index.idf:
            continue
        idf = index.idf[term]
        for doc_idx, tf in enumerate(term_frequencies):
            frequency = tf.get(term, 0)
            if frequency == 0:
                continue
            doc_length = index.doc_lengths[doc_idx]
            denominator = frequency + index.k1 * (
                1.0 - index.b + index.b * doc_length / index.avg_doc_length
            )
            scores[doc_idx] += idf * frequency * (index.k1 + 1.0) / denominator
    return scores


def test_bm25_inverted_index_matches_naive_reference() -> None:
    documents = [d.text for d in generate_corpus(n_docs=300, random_state=42)]
    index = BM25Index(documents)
    queries = [
        "rare unusual events in network traffic",
        "credit default probability risk scoring",
        "bm25 keyword matching term frequency",
        "genomic sequence matching across the genome",
        "a a a",  # repeated and stopword-ish tokens
        "tokenthatdoesnotexist",
    ]
    for query in queries:
        new = index.scores(query)
        ref = _reference_bm25_scores(index, query)
        # Bit-for-bit identical: same per-document accumulation order -> same floats.
        assert np.array_equal(new, ref), query
        # And the top-k rankings (with score tie-breaking) are identical.
        new_rank = [r.index for r in index.search(query, k=10)]
        ref_top = np.argpartition(-ref, kth=min(10, len(ref)) - 1)[:10]
        ref_rank = list(ref_top[np.argsort(-ref[ref_top])][: len(new_rank)])
        assert new_rank == [int(i) for i in ref_rank], query


# --------------------------------------------------------------------------- #
# B. Hybrid pipeline
# --------------------------------------------------------------------------- #
def _small_hybrid(reranker, candidate_count=20):
    documents = generate_corpus(n_docs=200, random_state=42)
    texts = [d.text for d in documents]
    embedder = SemanticHashingEmbedder(dim=64, random_state=42)
    vectors = embedder.encode(texts)
    bm25 = BM25Index(texts)
    return HybridSearcher(bm25, vectors, texts, embedder, reranker, candidate_count), bm25, documents


def test_hybrid_topk_is_a_reranking_of_bm25_candidates() -> None:
    searcher, bm25, _ = _small_hybrid(BiEncoderReranker(), candidate_count=20)
    query = "rare unusual events in network traffic"
    candidate_indices = {r.index for r in bm25.search(query, k=20)}
    results = searcher.search(query, k=10)

    assert len(results) == 10
    # Every hybrid result is one of the BM25 stage-1 candidates (the core invariant).
    assert all(r.index in candidate_indices for r in results)
    # Final ranks are contiguous and rerank scores are non-increasing.
    assert [r.final_rank for r in results] == list(range(1, 11))
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_fallback_reranker_is_deterministic() -> None:
    searcher, _, _ = _small_hybrid(CrossEncoderFallbackReranker(), candidate_count=20)
    first = [r.index for r in searcher.search("credit default risk scoring", k=10)]
    second = [r.index for r in searcher.search("credit default risk scoring", k=10)]
    assert first == second


def test_build_reranker_bi_encoder() -> None:
    reranker = build_reranker("bi-encoder")
    assert reranker.name == "bi-encoder"
    assert reranker.is_fallback is False


def test_reranker_metadata_flags_fallback() -> None:
    # The fallback must always self-report so it is never read as a real cross-encoder.
    fb = CrossEncoderFallbackReranker().metadata()
    assert fb["is_fallback"] is True
    assert fb["name"] == "cross-encoder-fallback"
    assert build_reranker("bi-encoder").metadata()["is_fallback"] is False


def test_build_reranker_auto_falls_back_when_unavailable(monkeypatch) -> None:
    import vector_search_memoire.hybrid as hyb

    monkeypatch.setattr(hyb, "cross_encoder_available", lambda: False)
    reranker = hyb.build_reranker("auto")
    assert reranker.is_fallback is True
    assert reranker.metadata()["is_fallback"] is True


@pytest.mark.skipif(not cross_encoder_available(), reason="sentence-transformers not installed")
def test_cross_encoder_reranker_scores_pairs() -> None:
    from vector_search_memoire.hybrid import CrossEncoderReranker

    try:
        reranker = CrossEncoderReranker()
    except Exception as exc:  # pragma: no cover - model download may fail offline
        pytest.skip(f"cross-encoder model unavailable: {exc}")
    scores = reranker.score(
        "patient triage in hospital",
        np.zeros(1, dtype=np.float32),
        [0, 1],
        ["hospital emergency patient triage routing", "solar wind energy forecast"],
        np.zeros((2, 1), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    assert scores.shape == (2,)
    # The medical document must score higher than the unrelated energy document.
    assert scores[0] > scores[1]


# --------------------------------------------------------------------------- #
# C. Scaling benchmark (quick smoke test)
# --------------------------------------------------------------------------- #
def test_make_unit_vectors_are_normalized_and_reproducible() -> None:
    a = make_unit_vectors(50, 16, seed=42)
    b = make_unit_vectors(50, 16, seed=42)
    assert np.allclose(a, b)
    norms = np.linalg.norm(a, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_benchmark_scale_smoke() -> None:
    rows = benchmark_scale(sizes=(200, 500), dim=32, n_queries=5, k=10, repeats=1)
    assert len(rows) == 2
    for row in rows:
        assert row["n_docs"] in (200, 500)
        assert 0.0 <= row["recall_at_k"] <= 1.0
        assert row["exact_query_ms"] > 0
        assert row["ann_query_ms"] > 0
        assert row["exact_mem_mb"] > 0
    assert crossover_size(rows) is None or crossover_size(rows) in (200, 500)
