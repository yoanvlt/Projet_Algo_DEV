"""Scaling benchmark: brute-force exact search vs Faiss HNSW.

The course claims HNSW gives "95-99% recall@10 with a 100-1000x speed-up" and an
``O(d log n)`` query complexity, but never demonstrates it; the project only ran at
2000 documents, where exact search is already instant. This module exhibits the
**crossover point** where HNSW starts to clearly beat brute force, and the cost
(recall, memory) paid for that speed.

For the timing/recall part we use *synthetic* vectors (fixed seed) rather than real
text: latency and recall depend on the number of vectors, the dimension and the
geometry of the space, not on the sentences behind them, so synthetic vectors at the
realistic dimension 384 (``all-MiniLM-L6-v2``) give the same curves while letting us
scale to 1M points reproducibly and offline.

Crucially the vectors are **clustered**, not uniform random. Real embeddings do not
fill the sphere uniformly: they concentrate on a lower-dimensional manifold with
topical clusters. Uniform Gaussian vectors are the pathological worst case for any
ANN index (every point is roughly equidistant, so recall collapses as ``n`` grows),
which would *understate* HNSW.

To isolate the effect of corpus **size**, the whole sweep uses a *single* generated
corpus with FIXED centroids, a FIXED number of clusters and a FIXED spread; each size
is a **prefix** of that corpus, and the query set is generated once and shared. Only
the number of vectors changes between sizes, so a recall change reflects volume/density
within a fixed distribution -- not a change of distribution (which an earlier version,
varying ``n_clusters`` with ``n``, accidentally conflated). The exact parameters are
recorded by :func:`generation_params`, and Python/NumPy/Faiss/hardware by
:func:`environment_info`; latencies are machine-dependent.
"""

from __future__ import annotations

import time

import numpy as np

from .vector_index import (
    ApproximateGraphIndex,
    ExactVectorIndex,
    FaissHNSWIndex,
    faiss_available,
)

QUICK_SIZES = (2_000, 10_000, 50_000)
FULL_SIZES = (2_000, 10_000, 50_000, 200_000, 1_000_000)
# The pure-Python graph fallback builds an O(n^2) similarity matrix, so it cannot
# scale; cap its corpus sizes when Faiss is unavailable.
FALLBACK_MAX_SIZE = 10_000


def _make_centroids(n_clusters: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    return centroids


def make_unit_vectors(
    n: int,
    dim: int,
    seed: int,
    n_clusters: int | None = None,
    spread: float = 0.08,
) -> np.ndarray:
    """Unit-normalised vectors drawn around topical centroids (see module docstring).

    ``n_clusters`` defaults to ``~ n / 1000`` (clamped to ``[8, 1000]``), mimicking a
    corpus whose number of topics grows slowly with its size. ``spread`` controls how
    tight the clusters are; the centroids use a fixed seed so the cluster *layout* is
    shared between the corpus and the query set, giving every query well-defined near
    neighbours. The defaults put recall@10 in the course's 95-99% regime.
    """

    if n_clusters is None:
        n_clusters = int(np.clip(n // 1_000, 8, 1_000))
    centroids = _make_centroids(n_clusters, dim, seed=10_000)  # shared layout
    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, n_clusters, size=n)
    noise = rng.normal(scale=spread, size=(n, dim)).astype(np.float32)
    vectors = centroids[assignments] + noise
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors.astype(np.float32)


# A fixed number of clusters is used for the WHOLE sweep so that only the corpus
# size changes between rows (see benchmark_scale): the data distribution is held
# constant and we draw more points from it, instead of changing the geometry.
DEFAULT_N_CLUSTERS = 256
DEFAULT_SPREAD = 0.08
# A speed-up only becomes operationally meaningful past this factor (vs the strict
# >1x crossover, which is noise-sensitive at small corpus sizes).
PRACTICAL_SPEEDUP = 5.0


def _latency_stats(index, query_vectors: np.ndarray, k: int, repeats: int) -> dict[str, float]:
    """Per-query latency stats in ms (mean / std / median), after one warm-up."""

    index.search(query_vectors[0], k=k)  # warm-up
    per_query: list[float] = []
    for query in query_vectors:
        start = time.perf_counter()
        for _ in range(repeats):
            index.search(query, k=k)
        per_query.append((time.perf_counter() - start) / repeats)
    arr = np.asarray(per_query, dtype=np.float64) * 1_000.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
    }


def _ann_index_bytes(index) -> int:
    """Serialized footprint of the ANN index (Faiss) or graph adjacency."""

    if isinstance(index, FaissHNSWIndex):
        import faiss

        return int(faiss.serialize_index(index.index).nbytes)
    if isinstance(index, ApproximateGraphIndex):
        return int(sum(neigh.nbytes for neigh in index.neighbors))
    return 0


def generation_params(
    dim: int = 384,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    spread: float = DEFAULT_SPREAD,
    n_queries: int = 20,
    random_state: int = 42,
    m: int = 32,
    ef_search: int = 80,
) -> dict[str, float | int]:
    """The exact generation/index parameters, recorded alongside the results so a
    recall figure cannot be misread as a pure size effect."""

    return {
        "dim": dim,
        "n_clusters": n_clusters,
        "spread": spread,
        "n_queries": n_queries,
        "random_state": random_state,
        "centroid_seed": 10_000,
        "hnsw_m": m,
        "hnsw_ef_search": ef_search,
        "distribution": "fixed clustered gaussians; sizes are prefixes of one max corpus",
    }


def environment_info() -> dict[str, str]:
    """Versions and hardware so latency numbers can be situated (machine-dependent)."""

    import platform

    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import faiss

        info["faiss"] = getattr(faiss, "__version__", "unknown")
    except Exception:
        info["faiss"] = "not installed"
    return info


def benchmark_vectors(
    vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int = 10,
    repeats: int = 5,
    m: int = 32,
    ef_search: int = 80,
    prefer_faiss: bool = True,
) -> dict[str, float | int | str]:
    """Benchmark exact vs ANN on already-generated vectors (one corpus prefix)."""

    exact = ExactVectorIndex(vectors)
    exact_neighbors = [
        {result.index for result in exact.search(query, k=k)} for query in query_vectors
    ]
    exact_lat = _latency_stats(exact, query_vectors, k, repeats)

    use_faiss = prefer_faiss and faiss_available()
    ann_build_start = time.perf_counter()
    if use_faiss:
        ann = FaissHNSWIndex(vectors, m=m, ef_search=ef_search)
        backend = "faiss-hnsw"
    else:
        ann = ApproximateGraphIndex(vectors)
        backend = "graph-ann-fallback"
    ann_build_ms = (time.perf_counter() - ann_build_start) * 1_000
    ann_lat = _latency_stats(ann, query_vectors, k, repeats)

    recalls = []
    for query, exact_set in zip(query_vectors, exact_neighbors):
        ann_set = {result.index for result in ann.search(query, k=k)}
        recalls.append(len(exact_set & ann_set) / k)

    return {
        "n_docs": len(vectors),
        "dim": int(vectors.shape[1]),
        "backend": backend,
        "exact_query_ms": exact_lat["mean"],
        "exact_query_ms_std": exact_lat["std"],
        "exact_query_ms_median": exact_lat["median"],
        "ann_query_ms": ann_lat["mean"],
        "ann_query_ms_std": ann_lat["std"],
        "ann_query_ms_median": ann_lat["median"],
        "speedup": exact_lat["mean"] / ann_lat["mean"] if ann_lat["mean"] > 0 else float("nan"),
        "recall_at_k": float(np.mean(recalls)),
        "ann_build_ms": ann_build_ms,
        "exact_mem_mb": vectors.nbytes / 1e6,
        "ann_mem_mb": _ann_index_bytes(ann) / 1e6,
        "k": k,
    }


def benchmark_scale(
    sizes: tuple[int, ...] | list[int] = QUICK_SIZES,
    dim: int = 384,
    n_queries: int = 20,
    k: int = 10,
    repeats: int = 5,
    m: int = 32,
    ef_search: int = 80,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    spread: float = DEFAULT_SPREAD,
    prefer_faiss: bool = True,
    random_state: int = 42,
) -> list[dict[str, float | int | str]]:
    """Benchmark every corpus size using **prefixes of a single max corpus**.

    The full corpus and the query set are generated once with fixed centroids,
    fixed ``n_clusters`` and fixed ``spread``; each size is the first ``n`` rows of
    that corpus. Only the number of vectors changes between rows, so a recall
    change is attributable to volume/density within a *fixed distribution* rather
    than to a distribution change (which the previous per-size generation conflated).
    """

    use_faiss = prefer_faiss and faiss_available()
    eligible = [n for n in sizes if use_faiss or n <= FALLBACK_MAX_SIZE]
    if not eligible:
        return []
    max_n = max(eligible)

    corpus = make_unit_vectors(
        max_n, dim, seed=random_state, n_clusters=n_clusters, spread=spread
    )
    query_vectors = make_unit_vectors(
        n_queries, dim, seed=random_state + 1, n_clusters=n_clusters, spread=spread
    )

    rows: list[dict[str, float | int | str]] = []
    for n_docs in sorted(eligible):
        rows.append(
            benchmark_vectors(
                corpus[:n_docs],
                query_vectors,
                k=k,
                repeats=repeats,
                m=m,
                ef_search=ef_search,
                prefer_faiss=prefer_faiss,
            )
        )
    return rows


def crossover_size(
    rows: list[dict[str, float | int | str]], min_speedup: float = 1.0
) -> int | None:
    """Smallest corpus size where HNSW reaches ``min_speedup`` over brute force.

    ``min_speedup=1.0`` is the *strict* crossover (HNSW merely faster, noise-sensitive
    at small sizes); pass :data:`PRACTICAL_SPEEDUP` for the *practical* crossover where
    the gain is operationally meaningful.
    """

    for row in sorted(rows, key=lambda r: r["n_docs"]):
        if float(row["speedup"]) >= min_speedup:
            return int(row["n_docs"])
    return None


def practical_crossover_size(
    rows: list[dict[str, float | int | str]], min_speedup: float = PRACTICAL_SPEEDUP
) -> int | None:
    """Smallest size where the speed-up is operationally significant (>= 5x default)."""

    return crossover_size(rows, min_speedup=min_speedup)
