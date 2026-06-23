from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "intro_embeddings"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


PHRASES = [
    ("semantic", "semantic search retrieves documents by meaning"),
    ("semantic", "dense retrieval finds passages with similar intent"),
    ("semantic", "vector search compares embeddings with cosine similarity"),
    ("semantic", "neural document retrieval handles paraphrased questions"),
    ("lexical", "BM25 ranks documents with term frequency and inverse document frequency"),
    ("lexical", "keyword search matches exact words in an inverted index"),
    ("lexical", "a rare query term receives a high IDF score"),
    ("lexical", "long documents are normalized in BM25 scoring"),
    ("medical", "radiology models detect anomalies in xray images"),
    ("medical", "clinical triage prioritizes patients in emergency departments"),
    ("medical", "hospital routing systems support patient care decisions"),
    ("medical", "scan interpretation can assist medical diagnosis"),
    ("finance", "fraud detection identifies suspicious payment transactions"),
    ("finance", "credit risk models estimate default probability"),
    ("finance", "transaction monitoring detects unusual banking behavior"),
    ("finance", "portfolio risk scoring ranks financial exposure"),
    ("robotics", "mobile robots plan paths around obstacles"),
    ("robotics", "sensor fusion combines camera and lidar signals"),
    ("robotics", "autonomous navigation chooses a safe route"),
    ("robotics", "warehouse robots optimize trajectories between shelves"),
]


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def save_heatmap(matrix: np.ndarray, phrase_ids: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, vmin=-0.1, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(phrase_ids)))
    ax.set_yticks(range(len(phrase_ids)))
    ax.set_xticklabels(phrase_ids, rotation=90)
    ax.set_yticklabels(phrase_ids)
    ax.set_title("Similarité cosinus entre 20 phrases")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cosine_similarity_heatmap.png", dpi=180)
    plt.close(fig)


def save_umap(vectors: np.ndarray, labels: list[str], phrase_ids: list[str]) -> None:
    reducer = UMAP(n_components=2, n_neighbors=5, min_dist=0.2, metric="cosine", random_state=42)
    coords = reducer.fit_transform(vectors)
    df = pd.DataFrame(
        {
            "phrase_id": phrase_ids,
            "label": labels,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    df.to_csv(OUTPUT_DIR / "umap_2d.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "semantic": "#d94f5c",
        "lexical": "#3876bf",
        "medical": "#2a9d70",
        "finance": "#a66a00",
        "robotics": "#6d5bd0",
    }
    for label in sorted(set(labels)):
        subset = df[df["label"] == label]
        ax.scatter(subset["x"], subset["y"], label=label, s=80, color=colors[label])
        for _, row in subset.iterrows():
            ax.text(row["x"] + 0.04, row["y"] + 0.04, row["phrase_id"], fontsize=8)
    ax.set_title("Projection UMAP 2D des embeddings de 20 phrases")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(title="Groupe")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "umap_2d.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in PHRASES]
    texts = [text for _, text in PHRASES]
    phrase_ids = [f"P{i:02d}" for i in range(1, len(PHRASES) + 1)]

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(np.float32)

    phrase_rows = pd.DataFrame({"phrase_id": phrase_ids, "label": labels, "text": texts})
    phrase_rows.to_csv(OUTPUT_DIR / "phrases.csv", index=False)

    matrix = cosine_matrix(vectors)
    pd.DataFrame(matrix, index=phrase_ids, columns=phrase_ids).to_csv(
        OUTPUT_DIR / "cosine_similarity_matrix.csv"
    )
    save_heatmap(matrix, phrase_ids)
    save_umap(vectors, labels, phrase_ids)

    nearest_rows = []
    for i, phrase_id in enumerate(phrase_ids):
        scores = matrix[i].copy()
        scores[i] = -np.inf
        best = int(np.argmax(scores))
        nearest_rows.append(
            {
                "phrase_id": phrase_id,
                "label": labels[i],
                "text": texts[i],
                "nearest_phrase_id": phrase_ids[best],
                "nearest_label": labels[best],
                "nearest_text": texts[best],
                "cosine_similarity": float(matrix[i, best]),
            }
        )
    pd.DataFrame(nearest_rows).to_csv(OUTPUT_DIR / "nearest_neighbors.csv", index=False)
    print(f"Intro embedding artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to build intro embedding artifacts: {exc}", file=sys.stderr)
        raise
