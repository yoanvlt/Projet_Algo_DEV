from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from make_hnsw_diagram import main as save_hnsw_diagram


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"


def save_precision_chart() -> None:
    df = pd.read_csv(OUTPUTS / "evaluation.csv")
    x = range(len(df))

    plt.figure(figsize=(12, 5))
    plt.bar([i - 0.2 for i in x], df["bm25_precision_at_k"], width=0.4, label="BM25")
    plt.bar([i + 0.2 for i in x], df["vector_precision_at_k"], width=0.4, label="Vectoriel")
    plt.xticks(list(x), df["query_id"], rotation=45, ha="right")
    plt.ylim(0, 1.08)
    plt.ylabel("Precision@K")
    plt.xlabel("Requete")
    plt.title("Precision@K par requete : BM25 vs recherche semantique")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "precision_at_k.png", dpi=180)
    plt.close()


def save_regime_chart() -> None:
    """Grouped bars of mean nDCG@K per regime, overall and per query style."""

    path = OUTPUTS / "regime_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    regimes = ["bm25", "dense", "hybrid_bi", "hybrid_cross"]
    regimes = [r for r in regimes if r in df["regime"].unique()]
    groups = ["overall"] + sorted(df["style"].unique())

    def mean_ndcg(regime: str, group: str) -> float:
        sub = df[df["regime"] == regime]
        if group != "overall":
            sub = sub[sub["style"] == group]
        return float(sub["ndcg_at_k"].mean())

    width = 0.8 / len(regimes)
    plt.figure(figsize=(10, 5.2))
    for i, regime in enumerate(regimes):
        offsets = [j + (i - (len(regimes) - 1) / 2) * width for j in range(len(groups))]
        plt.bar(offsets, [mean_ndcg(regime, g) for g in groups], width=width, label=regime)
    plt.xticks(range(len(groups)), groups)
    plt.ylabel("nDCG@K moyen")
    plt.xlabel("Type de requete")
    plt.title("Trois regimes : BM25 / dense / hybride (nDCG@K)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "regime_comparison.png", dpi=180)
    plt.close()


def save_dimensionality_chart() -> None:
    df = pd.read_csv(OUTPUTS / "dimensionality.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["dimension"],
        df["coefficient_of_variation"],
        marker="o",
        linewidth=2,
    )
    plt.xscale("log")
    plt.xlabel("Dimension de l'espace vectoriel (echelle log)")
    plt.ylabel("Coefficient de variation des distances")
    plt.title("Concentration des distances quand la dimension augmente")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "distance_concentration.png", dpi=180)
    plt.close()


def save_hnsw_tradeoff_chart() -> None:
    df = pd.read_csv(OUTPUTS / "hnsw_tradeoff.csv")

    fig, recall_axis = plt.subplots(figsize=(8.6, 5.2))
    time_axis = recall_axis.twinx()
    recall_line = recall_axis.plot(
        df["ef_search"],
        df["recall_at_5_vs_exact"],
        color="#0f766e",
        marker="o",
        linewidth=2.2,
        label="Rappel@5 vs exact",
    )
    time_line = time_axis.plot(
        df["ef_search"],
        df["mean_query_ms"],
        color="#ea580c",
        marker="s",
        linewidth=2.2,
        label="Temps moyen",
    )
    recall_axis.set_xscale("log", base=2)
    recall_axis.set_xticks(df["ef_search"])
    recall_axis.set_xticklabels([str(value) for value in df["ef_search"]])
    recall_axis.set_ylim(0.84, 1.01)
    recall_axis.set_xlabel("efSearch")
    recall_axis.set_ylabel("Rappel@5 par rapport a l'exact", color="#0f766e")
    time_axis.set_ylabel("Temps moyen par requete (ms)", color="#ea580c")
    recall_axis.grid(True, alpha=0.25)
    lines = recall_line + time_line
    recall_axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    plt.title("HNSW : compromis entre rappel et temps de recherche")
    fig.tight_layout()
    fig.savefig(FIGURES / "hnsw_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    save_precision_chart()
    save_regime_chart()
    save_dimensionality_chart()
    save_hnsw_tradeoff_chart()
    save_hnsw_diagram()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
