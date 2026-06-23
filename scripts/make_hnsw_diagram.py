from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT_ROOT / "outputs" / "figures"


def arrow(ax, start, end, color="#4c566a", width=1.2, alpha=0.85, style="-|>") -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=width,
        color=color,
        alpha=alpha,
        shrinkA=8,
        shrinkB=8,
    )
    ax.add_patch(patch)


def draw_layer(ax, y: float, nodes: dict[str, tuple[float, float]], active: list[str]) -> None:
    for name, (x, _) in nodes.items():
        face = "#d94f5c" if name in active else "#f5f7fb"
        edge = "#d94f5c" if name in active else "#7b8494"
        ax.scatter([x], [y], s=520, c=face, edgecolors=edge, linewidths=1.8, zorder=3)
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            color="white" if name in active else "#252a31",
            fontsize=10,
            fontweight="bold",
            zorder=4,
        )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 8)
    ax.axis("off")

    layers = {
        "Couche 2 : raccourcis longue distance": (6.7, {"A": (1.2, 6.7), "D": (6.2, 6.7), "H": (10.3, 6.7)}, ["A", "D"]),
        "Couche 1 : graphe plus dense": (
            4.3,
            {"A": (1.2, 4.3), "B": (3.0, 4.3), "D": (6.2, 4.3), "F": (8.1, 4.3), "H": (10.3, 4.3)},
            ["D", "F"],
        ),
        "Couche 0 : tous les vecteurs": (
            1.9,
            {
                "A": (0.8, 1.9),
                "B": (2.0, 1.9),
                "C": (3.2, 1.9),
                "D": (4.4, 1.9),
                "E": (5.6, 1.9),
                "F": (6.8, 1.9),
                "G": (8.0, 1.9),
                "H": (9.2, 1.9),
                "I": (10.4, 1.9),
            },
            ["F", "G"],
        ),
    }

    for label, (y, nodes, active) in layers.items():
        ax.text(0.3, y + 0.65, label, fontsize=12, fontweight="bold", color="#252a31")
        ax.hlines(y - 0.62, 0.3, 11.3, colors="#d8dde6", linestyles="dashed", linewidth=1)
        draw_layer(ax, y, nodes, active)

    # In-layer neighborhood links.
    arrow(ax, (1.2, 6.7), (6.2, 6.7), width=1.4)
    arrow(ax, (6.2, 6.7), (10.3, 6.7), width=1.1, alpha=0.45)

    for x1, x2 in [(1.2, 3.0), (3.0, 6.2), (6.2, 8.1), (8.1, 10.3)]:
        arrow(ax, (x1, 4.3), (x2, 4.3), width=1.1, alpha=0.55, style="-")

    for x1, x2 in [(0.8, 2.0), (2.0, 3.2), (3.2, 4.4), (4.4, 5.6), (5.6, 6.8), (6.8, 8.0), (8.0, 9.2), (9.2, 10.4)]:
        arrow(ax, (x1, 1.9), (x2, 1.9), width=0.9, alpha=0.5, style="-")

    # Search path.
    path_color = "#d94f5c"
    arrow(ax, (0.55, 7.55), (1.2, 6.7), color=path_color, width=2.0)
    arrow(ax, (1.2, 6.7), (6.2, 6.7), color=path_color, width=2.3)
    arrow(ax, (6.2, 6.45), (6.2, 4.55), color=path_color, width=2.3)
    arrow(ax, (6.2, 4.3), (8.1, 4.3), color=path_color, width=2.3)
    arrow(ax, (8.1, 4.05), (6.8, 2.15), color=path_color, width=2.3)
    arrow(ax, (6.8, 1.9), (8.0, 1.9), color=path_color, width=2.3)

    ax.text(0.35, 7.55, "Requête", fontsize=11, color=path_color, fontweight="bold")
    ax.text(7.55, 1.05, "voisins candidats (top K)", fontsize=11, color=path_color, fontweight="bold")

    ax.text(
        0.3,
        0.45,
        "Intuition : HNSW commence sur une couche peu dense pour se déplacer vite,\n"
        "puis descend vers une couche plus dense pour raffiner les voisins proches.",
        fontsize=10.5,
        color="#3a4250",
    )

    fig.tight_layout()
    fig.savefig(FIGURES / "hnsw_layers.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"HNSW diagram written to {FIGURES / 'hnsw_layers.png'}")


if __name__ == "__main__":
    main()
