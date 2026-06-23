"""Benchmark exact brute-force search vs Faiss HNSW across corpus sizes.

Writes ``outputs/scale_benchmark.csv`` and two figures into ``outputs/figures``:

- ``scale_crossover.png`` : query latency vs corpus size (log-log), showing where
  HNSW overtakes brute force;
- ``scale_recall_latency.png`` : the recall / latency / memory trade-off.

Examples (PowerShell)::

    .\.venv\Scripts\python.exe scripts\benchmark_scale.py --quick
    .\.venv\Scripts\python.exe scripts\benchmark_scale.py --full
    .\.venv\Scripts\python.exe scripts\benchmark_scale.py --sizes 2000 20000 100000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vector_search_memoire.scale import (  # noqa: E402
    DEFAULT_N_CLUSTERS,
    DEFAULT_SPREAD,
    FULL_SIZES,
    PRACTICAL_SPEEDUP,
    QUICK_SIZES,
    benchmark_scale,
    crossover_size,
    environment_info,
    generation_params,
    practical_crossover_size,
)
from vector_search_memoire.vector_index import faiss_available  # noqa: E402

OUTPUT_CSV = PROJECT_ROOT / "outputs" / "scale_benchmark.csv"
OUTPUT_META = PROJECT_ROOT / "outputs" / "scale_benchmark_meta.json"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - figures are optional
        print(f"matplotlib unavailable, skipping figures: {exc}")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sizes = [row["n_docs"] for row in rows]
    exact_ms = [row["exact_query_ms"] for row in rows]
    ann_ms = [row["ann_query_ms"] for row in rows]
    recall = [row["recall_at_k"] for row in rows]
    backend = rows[0]["backend"]

    # Figure 1: crossover (latency vs corpus size, log-log).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, exact_ms, "o-", label="Exact brute force", color="#c0392b")
    ax.plot(sizes, ann_ms, "s-", label=f"ANN ({backend})", color="#2471a3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Corpus size (number of vectors)")
    ax.set_ylabel("Mean query latency (ms)")
    strict = crossover_size(rows)
    practical = practical_crossover_size(rows)
    title = "HNSW vs brute force: query latency"
    if strict is not None:
        ax.axvline(strict, color="#b0b6bd", linestyle=":", linewidth=1,
                   label=f"strict crossover (>1x) ~ {strict:,}")
    if practical is not None:
        ax.axvline(practical, color="#7f8c8d", linestyle="--", linewidth=1.2,
                   label=f"practical (>={PRACTICAL_SPEEDUP:.0f}x) ~ {practical:,}")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "scale_crossover.png", dpi=150)
    plt.close(fig)

    # Figure 2: recall vs latency, point size = corpus size.
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.plot(ann_ms, recall, "-", color="#2471a3", alpha=0.5, zorder=1)
    scatter = ax.scatter(
        ann_ms,
        recall,
        c=sizes,
        cmap="viridis",
        s=80,
        norm=__import__("matplotlib").colors.LogNorm(),
        zorder=2,
    )
    # Stagger annotations so they do not overlap (alternate above/below each point).
    order = sorted(range(len(rows)), key=lambda i: ann_ms[i])
    for rank, i in enumerate(order):
        above = rank % 2 == 0
        ax.annotate(
            f"{rows[i]['n_docs']:,}",
            (ann_ms[i], recall[i]),
            textcoords="offset points",
            xytext=(0, 11 if above else -15),
            ha="center",
            va="bottom" if above else "top",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#aab0b6", lw=0.6),
        )
    ax.set_xlabel(f"ANN query latency (ms, {backend})")
    ax.set_ylabel(f"Recall@{rows[0]['k']} vs exact")
    ax.set_ylim(min(recall) - 0.08, 1.03)
    ax.set_title("Recall / latency trade-off of HNSW")
    fig.colorbar(scatter, label="Corpus size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "scale_recall_latency.png", dpi=150)
    plt.close(fig)
    print(f"Figures written to {FIGURE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Stop at 50k vectors (laptop-friendly).")
    parser.add_argument("--full", action="store_true", help="Go up to 1M vectors (heavy).")
    parser.add_argument("--sizes", type=int, nargs="+", help="Explicit corpus sizes.")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    if args.sizes:
        sizes = tuple(args.sizes)
    elif args.full:
        sizes = FULL_SIZES
    else:
        sizes = QUICK_SIZES  # default is the quick, laptop-friendly sweep

    if not faiss_available():
        print(
            "WARNING: Faiss is not installed -> using the pure-Python graph fallback. "
            "Sizes above the fallback cap are skipped and the numbers are NOT the "
            "official HNSW results."
        )

    print(f"Benchmarking sizes={sizes} dim={args.dim} (prefixes of one max corpus) ...")
    rows = benchmark_scale(
        sizes=sizes,
        dim=args.dim,
        n_queries=args.n_queries,
        k=args.k,
        repeats=args.repeats,
        n_clusters=DEFAULT_N_CLUSTERS,
        spread=DEFAULT_SPREAD,
        random_state=args.random_state,
    )
    if not rows:
        print("No sizes were benchmarked (fallback cap). Install faiss-cpu for the full sweep.")
        return

    write_csv(rows, OUTPUT_CSV)
    meta = {
        "generation_params": generation_params(
            dim=args.dim, n_queries=args.n_queries, random_state=args.random_state
        ),
        "environment": environment_info(),
        "note": "Latencies are wall-clock and machine-dependent; compare ratios, not absolutes.",
        "sizes": [int(r["n_docs"]) for r in rows],
        "repeats": args.repeats,
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Benchmark written to {OUTPUT_CSV}")
    print(f"Metadata (params + environment) written to {OUTPUT_META}")
    for row in rows:
        print(
            f"  n={row['n_docs']:>9,}  exact={row['exact_query_ms']:.3f}+-{row['exact_query_ms_std']:.3f}ms  "
            f"ann={row['ann_query_ms']:.3f}+-{row['ann_query_ms_std']:.3f}ms  speedup={row['speedup']:.1f}x  "
            f"recall@{row['k']}={row['recall_at_k']:.3f}  "
            f"raw={row['exact_mem_mb']:.1f}MB  index={row['ann_mem_mb']:.1f}MB"
        )
    strict = crossover_size(rows)
    practical = practical_crossover_size(rows)
    print(f"Strict crossover (HNSW > brute force): {strict}")
    print(f"Practical crossover (>= {PRACTICAL_SPEEDUP:.0f}x speed-up): {practical}")
    if not args.no_figures:
        make_figures(rows)


if __name__ == "__main__":
    main()
