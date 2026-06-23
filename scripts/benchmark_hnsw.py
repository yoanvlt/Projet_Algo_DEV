from __future__ import annotations

import csv
from pathlib import Path
import sys
import time

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vector_search_memoire.dataset import default_queries
from vector_search_memoire.embeddings import SentenceTransformerEmbedder
from vector_search_memoire.vector_index import ExactVectorIndex, FaissHNSWIndex


OUTPUT = PROJECT_ROOT / "outputs" / "hnsw_tradeoff.csv"
EF_SEARCH_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
K = 5
TIMING_REPEATS = 50


def benchmark_index(
    query_vectors: np.ndarray,
    exact_neighbors: list[set[int]],
    vectors: np.ndarray,
    ef_search: int,
) -> dict[str, float | int]:
    ann = FaissHNSWIndex(vectors, m=32, ef_search=ef_search)
    recalls: list[float] = []
    elapsed: list[float] = []

    for query_vector, exact_set in zip(query_vectors, exact_neighbors):
        ann.search(query_vector, k=K)
        start = time.perf_counter()
        for _ in range(TIMING_REPEATS):
            ann.search(query_vector, k=K)
        elapsed.append((time.perf_counter() - start) / TIMING_REPEATS)
        results = ann.search(query_vector, k=K)
        ann_set = {result.index for result in results}
        recalls.append(len(exact_set & ann_set) / K)

    serialized_bytes = int(faiss.serialize_index(ann.index).nbytes)
    return {
        "ef_search": ef_search,
        "recall_at_5_vs_exact": float(np.mean(recalls)),
        "mean_query_ms": float(np.mean(elapsed) * 1_000),
        "index_bytes": serialized_bytes,
    }


def main() -> None:
    vectors_path = PROJECT_ROOT / "outputs" / "embeddings.npy"
    if not vectors_path.exists():
        raise FileNotFoundError(
            "Run scripts/run_experiment.py with sentence-transformers before this benchmark."
        )

    vectors = np.load(vectors_path).astype(np.float32)
    queries = default_queries()
    embedder = SentenceTransformerEmbedder()
    query_vectors = embedder.encode([query.text for query in queries])
    exact = ExactVectorIndex(vectors)
    exact_neighbors = [
        {result.index for result in exact.search(query_vector, k=K)}
        for query_vector in query_vectors
    ]

    rows = [
        benchmark_index(query_vectors, exact_neighbors, vectors, ef_search)
        for ef_search in EF_SEARCH_VALUES
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dense vector bytes: {vectors.nbytes}")
    print(f"Benchmark written to {OUTPUT}")


if __name__ == "__main__":
    main()
