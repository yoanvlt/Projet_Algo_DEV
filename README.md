# Bases vectorielles et recherche semantique — Sujet 9

Projet M1 Algorithmique et ingenierie de donnees. Auteurs : Nemo MULLER et Yoan VIOLLET.

Le cours pose la theorie (BM25, embeddings, HNSW, pipeline hybride) ; ce depot la **mesure** sur un banc reproductible (`random_state=42`) et repond a deux questions :

1. le pipeline hybride **BM25 -> re-ranking** depasse-t-il BM25 seul et la recherche dense seule, et sur quelles requetes ?
2. a partir de quelle taille de corpus **HNSW** devient-il indispensable face a la force brute, et a quel cout en rappel / memoire ?

Les reponses chiffrees et leur analyse sont dans le memoire (`livrables/memoire_sujet9_bases_vectorielles.pdf`).

## Livrables

- `livrables/memoire_sujet9_bases_vectorielles.pdf` — memoire
- `livrables/presentation_sujet9_bases_vectorielles.pptx` — support de soutenance (14 slides)
- `livrables/script_oral_15min.md` — script oral 10 a 15 minutes

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-full.txt
```

`requirements-minimal.txt` suffit pour la demo hors ligne (fallback deterministe si `sentence-transformers` / Faiss sont absents).

## Commandes principales

```powershell
# Experience complete : BM25 / dense / hybride, metriques graduees (nDCG@10, MAP@50)
.\.venv\Scripts\python.exe scripts\run_experiment.py --embedding-provider sentence-transformers --k 10 --candidate-count 50

# Banc de passage a l'echelle (force brute vs HNSW, 2k -> 1M)
.\.venv\Scripts\python.exe scripts\benchmark_scale.py --full      # ou --quick

# Figures du memoire
.\.venv\Scripts\python.exe scripts\make_figures.py

# Demo interactive
.\.venv\Scripts\streamlit.exe run app.py
```

Reconstruire le PDF du mémoire : `scripts\build_memoire_pdf.py`. Tests : `pytest`.

## Structure

```text
app.py                       Demo Streamlit (hybride / passage a l'echelle / evaluation)
src/vector_search_memoire/   Code principal (bm25, embeddings, vector_index, hybrid, scale, evaluation)
scripts/                     Experiences, benchmarks et export des livrables
outputs/                     Resultats et figures regeneres
memoire/                     Sources Markdown du memoire et annexe
livrables/                   PDF, PPTX et script oral
consignes/                   Sujet et guide fournis au depart
tests/                       Tests (pytest)
```
