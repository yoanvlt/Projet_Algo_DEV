from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover - Streamlit can still run without Plotly.
    px = None

from vector_search_memoire.bm25 import BM25Index
from vector_search_memoire.dataset import Query
from vector_search_memoire.evaluation import (
    compare_ann_recall,
    dimensionality_concentration,
)
from vector_search_memoire.experiments import (
    build_artifacts,
    build_hybrid_searcher,
    evaluate_regimes,
)
from vector_search_memoire.scale import (
    PRACTICAL_SPEEDUP,
    QUICK_SIZES,
    benchmark_scale,
    crossover_size,
    make_unit_vectors,
    practical_crossover_size,
)
from vector_search_memoire.vector_index import ExactVectorIndex, FaissHNSWIndex, faiss_available


st.set_page_config(
    page_title="Bases vectorielles - recherche semantique",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; max-width: 1320px;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    .result-card {background:#ffffff; border:1px solid #e6e9ee; border-radius:8px; padding:.5rem .7rem; margin-bottom:.5rem;}
    .result-rank {font-size: 0.74rem; color: #5b6472; text-transform: uppercase; letter-spacing: .04em;}
    .result-title {font-size: .98rem; font-weight: 650; margin-bottom: .12rem; color:#1f2733;}
    .result-meta {color: #526070; font-size: .82rem; margin-bottom: .25rem;}
    .result-text {color: #252a31; line-height: 1.4; font-size: .88rem;}
    .pipe-card {border:1px solid #e3e6ea; border-radius:8px; padding:.5rem .65rem; margin-bottom:.45rem; background:#ffffff; color:#252a31;}
    .pipe-up {border-left:4px solid #1e8e3e; background:#f3fbf5;}
    .pipe-down {border-left:4px solid #c0392b; background:#fdf4f3;}
    .pipe-flat {border-left:4px solid #b9c0c8; background:#fafbfc;}
    .pipe-new {border-left:4px solid #1565c0; background:#f0f6fd;}
    .badge {font-size:.72rem; font-weight:700; padding:.05rem .4rem; border-radius:10px;}
    .badge-up {color:#1e8e3e;} .badge-down {color:#c0392b;}
    .badge-flat {color:#7a828b;} .badge-new {color:#1565c0;}
    .dropped {opacity:.42;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached builders (so the demo stays fluid during the oral)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Construction du corpus et des index...")
def load_artifacts(n_docs: int, provider: str, random_state: int):
    return build_artifacts(n_docs=n_docs, random_state=random_state, embedding_provider=provider)


@st.cache_resource(show_spinner="Chargement du re-ranker...")
def load_hybrid(n_docs: int, provider: str, random_state: int, kind: str, candidate_count: int):
    artifacts = load_artifacts(n_docs, provider, random_state)
    return build_hybrid_searcher(artifacts, kind, candidate_count)


@st.cache_data(show_spinner="Évaluation des 3 régimes...")
def cached_regimes(n_docs: int, provider: str, random_state: int, k: int, candidate_count: int):
    artifacts = load_artifacts(n_docs, provider, random_state)
    searchers = {
        "hybrid_bi": load_hybrid(n_docs, provider, random_state, "bi-encoder", candidate_count),
        "hybrid_cross": load_hybrid(n_docs, provider, random_state, "auto", candidate_count),
    }
    rows, summary = evaluate_regimes(
        artifacts["documents"],
        artifacts["queries"],
        artifacts["bm25"],
        artifacts["exact_index"],
        artifacts["embedder"],
        searchers,
        k=k,
    )
    return rows, summary


@st.cache_data(show_spinner="Benchmark du passage à l'échelle...")
def cached_scale(sizes: tuple[int, ...], dim: int, k: int, repeats: int, random_state: int):
    return benchmark_scale(
        sizes=list(sizes), dim=dim, k=k, repeats=repeats, random_state=random_state
    )


@st.cache_data(show_spinner=False)
def load_precomputed_scale():
    path = PROJECT_ROOT / "outputs" / "scale_benchmark.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_scale_meta():
    import json

    path = PROJECT_ROOT / "outputs" / "scale_benchmark_meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner="Balayage efSearch...")
def ef_sweep(n_docs_ef: int, ef_values: tuple[int, ...], seed: int):
    """Recall/latency of HNSW vs efSearch at a fixed corpus size (cached)."""

    if not faiss_available():
        return None
    import time as _t

    n_clusters = int(np.clip(n_docs_ef // 1000, 8, 1000))
    vectors = make_unit_vectors(n_docs_ef, 384, seed=seed, n_clusters=n_clusters)
    queries_v = make_unit_vectors(20, 384, seed=seed + 1, n_clusters=n_clusters)
    exact = ExactVectorIndex(vectors)
    gold = [{r.index for r in exact.search(q, 10)} for q in queries_v]
    rows = []
    for ef in ef_values:
        idx = FaissHNSWIndex(vectors, m=32, ef_search=ef)
        recs, times = [], []
        for q, g in zip(queries_v, gold):
            idx.search(q, 10)
            start = _t.perf_counter()
            for _ in range(5):
                idx.search(q, 10)
            times.append((_t.perf_counter() - start) / 5 * 1000)
            a = {r.index for r in idx.search(q, 10)}
            recs.append(len(g & a) / 10)
        rows.append(
            {"efSearch": ef, "recall_at_10": float(np.mean(recs)), "latence_ms": float(np.mean(times))}
        )
    return pd.DataFrame(rows)


def render_result(document, score: float, rank: int) -> None:
    concepts = ", ".join(document.concepts)
    st.markdown(
        f"""
        <div class="result-card">
        <div class="result-rank">#{rank} · score {score:.3f}</div>
        <div class="result-title">{document.title}</div>
        <div class="result-meta">{document.doc_id} · {document.topic_label} · {concepts}</div>
        <div class="result-text">{document.abstract}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Paramètres")
    n_docs = st.slider("Documents", min_value=500, max_value=3_000, value=2_000, step=250)
    k = st.slider("Top K", min_value=3, max_value=10, value=10, step=1)
    candidate_count = st.slider("Candidats BM25 (N)", min_value=20, max_value=100, value=50, step=10)
    provider = st.selectbox(
        "Embeddings",
        options=["auto", "fallback", "sentence-transformers"],
        index=0,
        help="Auto utilise sentence-transformers si disponible, sinon le fallback reproductible.",
    )
    random_state = int(st.number_input("Graine", min_value=0, value=42, step=1))

artifacts = load_artifacts(n_docs, provider, random_state)
documents = artifacts["documents"]
queries = artifacts["queries"]
bm25 = artifacts["bm25"]
embedder = artifacts["embedder"]
exact_index = artifacts["exact_index"]
ann_index = artifacts["ann_index"]

assert isinstance(bm25, BM25Index)
assert isinstance(exact_index, ExactVectorIndex)

st.title("Bases vectorielles et recherche sémantique")
st.caption(
    "Ce que le cours affirme mais ne mesure pas : pipeline hybride + re-ranking, "
    "et passage à l'échelle de HNSW."
)

hybrid_tab, scale_tab, evaluation_tab, search_tab = st.tabs(
    [
        "🔀 Hybride / re-ranking",
        "📈 Passage à l'échelle / ANN",
        "📊 Évaluation (3 régimes)",
        "🔎 Recherche côte à côte",
    ]
)

# --------------------------------------------------------------------------- #
# Tab 1 - Hybrid / re-ranking (signature demo)
# --------------------------------------------------------------------------- #
with hybrid_tab:
    st.subheader("Pipeline en deux étages : BM25 (recall) → re-ranking → top-K")
    controls = st.columns([0.42, 0.30, 0.28])
    with controls[0]:
        query_mode = st.radio("Requête", ["Requêtes annotées", "Libre"], horizontal=True, key="hy_mode")
        if query_mode == "Requêtes annotées":
            selected = st.selectbox(
                "Choisir",
                options=queries,
                format_func=lambda q: f"{q.query_id} · [{q.style}] {q.text}",
                key="hy_query",
            )
            query_text = selected.text
            st.caption(selected.note or "Requête étiquetée.")
        else:
            query_text = st.text_input("Texte", value="grouping users into cohorts from sessions", key="hy_text")
    with controls[1]:
        reranker_choice = st.radio(
            "Re-ranker",
            ["cross-encoder", "bi-encoder"],
            help="Cross-encoder = score conjoint (requête, doc). Bi-encoder = cosinus des embeddings.",
            key="hy_rr",
        )
    with controls[2]:
        st.metric("Candidats BM25", candidate_count)

    kind = "auto" if reranker_choice == "cross-encoder" else "bi-encoder"
    hybrid = load_hybrid(n_docs, provider, random_state, kind, candidate_count)
    if reranker_choice == "cross-encoder" and getattr(hybrid.reranker, "is_fallback", False):
        st.warning(
            "Cross-encoder indisponible : fallback déterministe utilisé (plan B). "
            "Les chiffres principaux exigent le vrai modèle.",
            icon="⚠️",
        )

    candidates = bm25.search(query_text, k=candidate_count)
    hybrid_results = hybrid.search(query_text, k=k)
    final_indices = {r.index for r in hybrid_results}

    col_cand, col_final, col_read = st.columns([0.36, 0.40, 0.24], gap="medium")
    with col_cand:
        st.markdown(f"**1 · BM25 — {len(candidates)} candidats**")
        for rank, result in enumerate(candidates[:12], start=1):
            doc = documents[result.index]
            survived = result.index in final_indices
            cls = "" if survived else "dropped"
            st.markdown(
                f"<div class='pipe-card pipe-flat {cls}'>"
                f"<span class='result-rank'>#{rank}</span> "
                f"<b>{doc.doc_id}</b> · {doc.topic_label}<br>"
                f"<span class='result-text'>{doc.title}</span></div>",
                unsafe_allow_html=True,
            )

    with col_final:
        st.markdown(f"**2 · Re-ranking ({hybrid.reranker.name}) → top-{k}**")
        for result in hybrid_results:
            doc = documents[result.index]
            delta = result.rank_delta
            if result.stage1_rank > k:
                cls, bcls, badge = "pipe-new", "badge-new", f"⤴ entré (était #{result.stage1_rank})"
            elif delta > 0:
                cls, bcls, badge = "pipe-up", "badge-up", f"↑ +{delta}"
            elif delta < 0:
                cls, bcls, badge = "pipe-down", "badge-down", f"↓ {delta}"
            else:
                cls, bcls, badge = "pipe-flat", "badge-flat", "= inchangé"
            st.markdown(
                f"<div class='pipe-card {cls}'>"
                f"<span class='result-rank'>#{result.final_rank}</span> "
                f"<b>{doc.doc_id}</b> · {doc.topic_label} "
                f"<span class='badge {bcls}'>{badge}</span><br>"
                f"<span class='result-text'>{doc.title}</span></div>",
                unsafe_allow_html=True,
            )

    with col_read:
        st.markdown("**3 · Ce que le re-ranking a changé**")
        promoted = [r for r in hybrid_results if r.stage1_rank > k]
        moved = [r for r in hybrid_results if r.rank_delta != 0 and r.stage1_rank <= k]
        st.metric("Docs entrés dans le top-K", len(promoted))
        st.metric("Docs déplacés", len(moved))
        st.caption(
            "BM25 filtre largement (rappel), le re-ranker réordonne finement par sens. "
            "Les cartes bleues étaient hors du top-K avant re-ranking."
        )

# --------------------------------------------------------------------------- #
# Tab 2 - Scale / ANN
# --------------------------------------------------------------------------- #
with scale_tab:
    st.subheader("Point de bascule : force brute exacte vs Faiss HNSW")
    precomputed = load_precomputed_scale()
    scale_meta = load_scale_meta()
    if precomputed is not None:
        scale_df = precomputed
        params = (scale_meta or {}).get("generation_params", {})
        seed_txt = params.get("random_state", "?")
        st.info(
            f"📦 **Courbe pré-calculée** (figée) depuis `outputs/scale_benchmark.csv` — "
            f"graine **{seed_txt}**, dim {params.get('dim', '?')}, "
            f"{params.get('n_clusters', '?')} clusters, efSearch {params.get('hnsw_ef_search', '?')}, "
            f"préfixes d'un corpus unique. Le slider « graine » de la barre latérale **n'affecte pas** "
            f"ce CSV (seulement le balayage efSearch en direct plus bas). "
            f"Régénérer : `scripts/benchmark_scale.py --full`.",
            icon="📦",
        )
    else:
        st.caption("Pas de CSV pré-calculé : benchmark rapide en direct (jusqu'à 50k).")
        scale_df = pd.DataFrame(cached_scale(QUICK_SIZES, 384, 10, 5, 42))

    records = scale_df.to_dict("records")
    strict = crossover_size(records)
    practical = practical_crossover_size(records)
    big = scale_df.sort_values("n_docs").iloc[-1]
    cols = st.columns(4)
    cols[0].metric(
        f"Bascule pratique (≥{PRACTICAL_SPEEDUP:.0f}×)",
        f"{practical:,}".replace(",", " ") if practical else "n/a",
        help=f"Bascule stricte (>1×) : {strict if strict else 'n/a'}. "
        "La bascule stricte est sensible au bruit aux petites tailles ; la pratique est le seuil utile.",
    )
    cols[1].metric(f"Speedup @ {int(big['n_docs']):,}".replace(",", " "), f"{big['speedup']:.0f}×")
    cols[2].metric(f"Recall@{int(big['k'])} @ max", f"{big['recall_at_k']:.3f}")
    cols[3].metric("Mémoire (brut → index)", f"{big['exact_mem_mb']:.0f} → {big['ann_mem_mb']:.0f} Mo")

    if px is not None:
        long = scale_df.melt(
            id_vars=["n_docs"],
            value_vars=["exact_query_ms", "ann_query_ms"],
            var_name="moteur",
            value_name="latence_ms",
        ).replace({"exact_query_ms": "Exact (force brute)", "ann_query_ms": "HNSW (ANN)"})
        fig = px.line(
            long, x="n_docs", y="latence_ms", color="moteur", markers=True, log_x=True, log_y=True,
            labels={"n_docs": "Taille du corpus", "latence_ms": "Latence requête (ms)"},
            height=380,
        )
        if strict:
            fig.add_vline(x=strict, line_dash="dot", line_color="#b0b6bd",
                          annotation_text="stricte", annotation_position="top")
        if practical:
            fig.add_vline(x=practical, line_dash="dash", line_color="#7f8c8d",
                          annotation_text=f"pratique ≥{PRACTICAL_SPEEDUP:.0f}×", annotation_position="top")
        st.plotly_chart(fig, width="stretch")

        fig2 = px.scatter(
            scale_df, x="ann_query_ms", y="recall_at_k", size="n_docs", color="n_docs",
            labels={"ann_query_ms": "Latence HNSW (ms)", "recall_at_k": f"Recall@{int(big['k'])}"},
            height=340, color_continuous_scale="viridis",
        )
        st.plotly_chart(fig2, width="stretch")
    st.dataframe(scale_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("**Contrôle `efSearch` en direct** (recalculé, recall vs latence à taille fixe)")
    ef_cols = st.columns([0.5, 0.5])
    with ef_cols[0]:
        ef_n = st.select_slider("Taille du corpus (live)", options=[2_000, 10_000, 50_000], value=10_000)
    with ef_cols[1]:
        st.caption(
            f"Calculé en direct avec la graine **{random_state}** de la barre latérale "
            "(distinct du CSV pré-calculé ci-dessus). efSearch ↑ ⇒ rappel ↑ mais latence ↑."
        )

    ef_df = ef_sweep(int(ef_n), (8, 16, 32, 64, 128, 256), random_state)
    if ef_df is None:
        st.info("Faiss absent : balayage efSearch indisponible (fallback graphe utilisé ailleurs).")
    elif px is not None:
        fig3 = px.line(
            ef_df, x="latence_ms", y="recall_at_10", markers=True, text="efSearch",
            labels={"latence_ms": "Latence (ms)", "recall_at_10": "Recall@10"}, height=320,
        )
        fig3.update_traces(textposition="top center")
        st.plotly_chart(fig3, width="stretch")
    else:
        st.dataframe(ef_df, width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 3 - Evaluation (3 regimes + new metrics)
# --------------------------------------------------------------------------- #
with evaluation_tab:
    rows, summary = cached_regimes(n_docs, provider, random_state, k, candidate_count)
    regime_df = pd.DataFrame(rows)
    means = (
        regime_df.groupby("regime")[["precision_at_k", "recall_at_k", "ndcg_at_k", "average_precision"]]
        .mean()
        .rename(columns={"average_precision": "MAP", "ndcg_at_k": f"nDCG@{k}",
                         "precision_at_k": f"P@{k}", "recall_at_k": f"R@{k}"})
    )
    st.subheader("Moyennes par régime")
    st.dataframe(means.style.format("{:.3f}"), width="stretch")

    verdict = summary.get("conclusion", {}).get("verdict", "")
    if verdict:
        st.info(verdict, icon="🧪")

    if px is not None:
        style_rows = []
        for style, d in summary["ndcg_by_style"].items():
            for regime, val in d.items():
                style_rows.append({"style": style, "regime": regime, f"nDCG@{k}": val})
        style_df = pd.DataFrame(style_rows)
        fig = px.bar(
            style_df, x="style", y=f"nDCG@{k}", color="regime", barmode="group", height=380,
            labels={"style": "Type de requête"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "L'hybride dépasse BM25 ET le dense surtout sur les requêtes sémantiques "
            "(paraphrases) — là où BM25 filtre par mots-clés et le re-ranker corrige le sens."
        )

    st.subheader("Détail par requête")
    st.dataframe(regime_df, width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 4 - Legacy side-by-side
# --------------------------------------------------------------------------- #
with search_tab:
    left, right = st.columns([0.4, 0.6], gap="large")
    with left:
        sb_mode = st.radio("Source", ["Requêtes annotées", "Libre"], horizontal=True, key="sb_mode")
        if sb_mode == "Requêtes annotées":
            sb_selected = st.selectbox(
                "Requête", options=queries,
                format_func=lambda q: f"{q.query_id} · {q.text}", key="sb_query",
            )
            sb_text = sb_selected.text
        else:
            sb_text = st.text_input("Requête", value="meaning based lookup for documents", key="sb_text")
        st.write("BM25 favorise les mots exacts ; le vectoriel les formulations proches en sens.")

    with right:
        sb_vec = embedder.encode([sb_text])[0]
        bm25_col, vector_col = st.columns(2, gap="large")
        with bm25_col:
            st.subheader("BM25")
            for rank, result in enumerate(bm25.search(sb_text, k=k), start=1):
                render_result(documents[result.index], result.score, rank)
        with vector_col:
            st.subheader("Recherche sémantique")
            for rank, result in enumerate(exact_index.search(sb_vec, k=k), start=1):
                render_result(documents[result.index], result.score, rank)

    with st.expander("ANN et concentration des distances"):
        ann_summary = compare_ann_recall(queries, embedder, exact_index, ann_index, k=k)
        cols = st.columns(3)
        cols[0].metric("Rappel ANN vs exact", f"{ann_summary['ann_recall_at_k_vs_exact']:.2f}")
        cols[1].metric("Temps exact moyen", f"{ann_summary['exact_mean_seconds'] * 1_000:.3f} ms")
        cols[2].metric("Temps ANN moyen", f"{ann_summary['ann_mean_seconds'] * 1_000:.3f} ms")
        st.dataframe(pd.DataFrame(dimensionality_concentration()), width="stretch", hide_index=True)
