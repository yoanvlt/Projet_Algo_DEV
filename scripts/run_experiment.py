from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vector_search_memoire.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the subject 9 search experiment.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-docs", type=int, default=2_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--embedding-provider",
        choices=["auto", "fallback", "sentence-transformers"],
        default="auto",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-K for the official metrics.")
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=50,
        help="BM25 stage-1 top-N feeding the hybrid re-ranker.",
    )
    args = parser.parse_args()

    summary = run_experiment(
        output_dir=args.output_dir,
        n_docs=args.n_docs,
        random_state=args.random_state,
        embedding_provider=args.embedding_provider,
        k=args.k,
        candidate_count=args.candidate_count,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
