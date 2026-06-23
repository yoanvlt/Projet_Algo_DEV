"""Exact and approximate vector search indexes."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import importlib.util
import time

import numpy as np


@dataclass(frozen=True)
class VectorSearchResult:
    index: int
    score: float


def top_k_from_scores(scores: np.ndarray, k: int) -> list[VectorSearchResult]:
    if len(scores) == 0:
        return []
    k = min(k, len(scores))
    candidate_indices = np.argpartition(-scores, kth=k - 1)[:k]
    ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
    return [VectorSearchResult(index=int(idx), score=float(scores[idx])) for idx in ordered]


class ExactVectorIndex:
    """Brute-force cosine search over normalized vectors."""

    name = "exact-cosine"

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype(np.float32)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[VectorSearchResult]:
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores = (self.vectors @ query.T).ravel()
        return top_k_from_scores(scores, k)


class FaissHNSWIndex:
    """Faiss HNSW wrapper for cosine similarity on normalized vectors."""

    name = "faiss-hnsw"

    def __init__(self, vectors: np.ndarray, m: int = 32, ef_search: int = 80) -> None:
        import faiss

        self.vectors = vectors.astype(np.float32)
        self.dim = self.vectors.shape[1]
        try:
            self.index = faiss.IndexHNSWFlat(self.dim, m, faiss.METRIC_INNER_PRODUCT)
            self.metric = "inner_product"
        except TypeError:
            self.index = faiss.IndexHNSWFlat(self.dim, m)
            self.metric = "l2"
        self.index.hnsw.efSearch = ef_search
        self.index.add(self.vectors)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[VectorSearchResult]:
        query = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, k)
        results: list[VectorSearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            if self.metric == "l2":
                score = 1.0 - float(score) / 2.0
            results.append(VectorSearchResult(index=int(idx), score=float(score)))
        return results


class ApproximateGraphIndex:
    """One-layer graph ANN fallback inspired by HNSW search.

    This is not a full HNSW implementation. It keeps the oral-demo intuition:
    search visits a limited candidate set by expanding neighbors that look closer
    to the query, instead of scoring every vector.
    """

    name = "graph-ann-fallback"

    def __init__(
        self,
        vectors: np.ndarray,
        neighbor_count: int = 24,
        ef_search: int = 160,
        entry_points: int = 8,
    ) -> None:
        self.vectors = vectors.astype(np.float32)
        self.neighbor_count = neighbor_count
        self.ef_search = ef_search
        self.entry_points = entry_points
        self.neighbors = self._build_graph()

    def _build_graph(self) -> list[np.ndarray]:
        n_vectors = len(self.vectors)
        if n_vectors == 0:
            return []
        similarities = self.vectors @ self.vectors.T
        np.fill_diagonal(similarities, -np.inf)
        neighbors: list[np.ndarray] = []
        k = min(self.neighbor_count, max(1, n_vectors - 1))
        for row in similarities:
            candidate_indices = np.argpartition(-row, kth=k - 1)[:k]
            ordered = candidate_indices[np.argsort(-row[candidate_indices])]
            neighbors.append(ordered.astype(np.int32))
        return neighbors

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[VectorSearchResult]:
        if len(self.vectors) == 0:
            return []

        query = query_vector.reshape(-1).astype(np.float32)
        n_vectors = len(self.vectors)
        entry_count = min(self.entry_points, n_vectors)
        entry_indices = np.linspace(0, n_vectors - 1, num=entry_count, dtype=np.int32)

        visited: set[int] = set()
        frontier: list[tuple[float, int]] = []
        for idx in entry_indices:
            score = float(self.vectors[int(idx)] @ query)
            heapq.heappush(frontier, (-score, int(idx)))
            visited.add(int(idx))

        while frontier and len(visited) < min(self.ef_search, n_vectors):
            _, current = heapq.heappop(frontier)
            for neighbor in self.neighbors[current]:
                neighbor_idx = int(neighbor)
                if neighbor_idx in visited:
                    continue
                visited.add(neighbor_idx)
                score = float(self.vectors[neighbor_idx] @ query)
                heapq.heappush(frontier, (-score, neighbor_idx))
                if len(visited) >= min(self.ef_search, n_vectors):
                    break

        visited_indices = np.fromiter(visited, dtype=np.int32)
        scores = self.vectors[visited_indices] @ query
        k = min(k, len(visited_indices))
        candidate_indices = np.argpartition(-scores, kth=k - 1)[:k]
        ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
        return [
            VectorSearchResult(index=int(visited_indices[pos]), score=float(scores[pos]))
            for pos in ordered
        ]


def faiss_available() -> bool:
    return importlib.util.find_spec("faiss") is not None


def build_ann_index(vectors: np.ndarray, prefer_faiss: bool = True):
    start = time.perf_counter()
    if prefer_faiss and faiss_available():
        index = FaissHNSWIndex(vectors)
    else:
        index = ApproximateGraphIndex(vectors)
    build_seconds = time.perf_counter() - start
    return index, build_seconds
