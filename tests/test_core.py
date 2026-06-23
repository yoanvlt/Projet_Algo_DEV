import math

import numpy as np
import pytest

from vector_search_memoire.bm25 import BM25Index
from vector_search_memoire.dataset import default_queries, generate_corpus
from vector_search_memoire.embeddings import SemanticHashingEmbedder
from vector_search_memoire.evaluation import evaluate_all, precision_recall_at_k
from vector_search_memoire.vector_index import ExactVectorIndex, FaissHNSWIndex, faiss_available


def test_corpus_is_reproducible() -> None:
    first = generate_corpus(n_docs=20, random_state=42)
    second = generate_corpus(n_docs=20, random_state=42)
    assert [doc.to_dict() for doc in first] == [doc.to_dict() for doc in second]


def test_bm25_returns_ranked_results() -> None:
    documents = ["semantic retrieval for documents", "solar energy forecast"]
    index = BM25Index(documents)
    results = index.search("semantic documents", k=2)
    assert results[0].index == 0
    assert results[0].score >= results[1].score


def test_bm25_matches_the_documented_formula() -> None:
    index = BM25Index(["alpha alpha beta", "beta gamma"], k1=1.5, b=0.75)
    score = float(index.scores("alpha")[0])
    expected_idf = math.log(1.0 + (2 - 1 + 0.5) / (1 + 0.5))
    expected_denominator = 2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2.5)
    expected = expected_idf * 2 * (1.5 + 1) / expected_denominator
    assert score == pytest.approx(expected, rel=1e-6)


def test_precision_recall_at_k() -> None:
    metrics = precision_recall_at_k([4, 2, 9, 1], {1, 2, 3}, k=4)
    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == pytest.approx(2 / 3)
    assert metrics.relevant_found == 2


def test_vector_pipeline_runs() -> None:
    documents = generate_corpus(n_docs=80, random_state=42)
    queries = default_queries()[:3]
    texts = [document.text for document in documents]
    embedder = SemanticHashingEmbedder(dim=64, random_state=42)
    vectors = embedder.encode(texts)
    rows = evaluate_all(
        documents=documents,
        queries=queries,
        bm25=BM25Index(texts),
        exact_index=ExactVectorIndex(vectors),
        embedder=embedder,
        k=5,
    )
    assert len(rows) == 3
    assert all("vector_precision_at_k" in row for row in rows)


@pytest.mark.skipif(not faiss_available(), reason="Faiss is not installed")
def test_faiss_hnsw_recovers_exact_neighbors() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(120, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[17]
    exact = ExactVectorIndex(vectors).search(query, k=5)
    ann = FaissHNSWIndex(vectors, m=16, ef_search=64).search(query, k=5)
    assert {result.index for result in ann} == {result.index for result in exact}
