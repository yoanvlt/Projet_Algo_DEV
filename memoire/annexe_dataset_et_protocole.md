# Annexe - Dataset, pertinence graduée, protocole et listings

## Objectif du dataset

Le dataset est un **banc d'essai contrôlé** : il n'imite pas un corpus scientifique réel, il crée un environnement où l'on connaît exactement quels documents sont pertinents pour chaque requête. Cette vérité-terrain permet de calculer `precision@10`, `recall@10`, `nDCG@10` et `MAP@50` de façon reproductible.

## Génération du corpus

2 000 documents générés avec `random_state=42`. Chaque document possède : `doc_id`, `title`, `topic` (domaine), `topic_label`, `concepts` (deux concepts contrôlés, dont le premier est le **concept principal** rendu dans le titre), `domain` (terme de domaine précis), `abstract` et `text` (titre + résumé). Les huit domaines sont : `healthcare_ai`, `climate_energy`, `cybersecurity`, `finance_risk`, `recommender_systems`, `robotics`, `bioinformatics`, `legal_tech`. Les concepts incluent `semantic_search`, `bm25`, `anomaly_detection`, `cyber_threat`, `fraud_detection`, `risk_scoring`, `graph_recommendation`, `user_clustering`, `dna_alignment`, `document_classification`, `citation_retrieval`, `ranking`, etc.

## Pertinence graduée 0-3

La pertinence binaire initiale (« le document contient le concept demandé ») était trop large : certaines requêtes avaient jusqu'à plusieurs centaines de documents pertinents, ce qui saturait `precision@5` à 1,00. Nous l'avons remplacée par une pertinence **graduée**, fondée sur quatre facettes : le **domaine** (topic), la présence du **concept** cible, le fait que ce concept soit le **concept principal** (`concepts[0]`, celui du titre) et la correspondance du **terme de domaine** précis.

```text
grade 3  : topic correct ET concept principal correct ET terme de domaine correct   (le "gold")
grade 2  : topic correct ET concept principal correct, mais terme de domaine différent
grade 1  : thématiquement adjacent (concept présent mais secondaire dans le bon topic,
           ou concept principal d'un autre topic)
grade 0  : non pertinent
```

La pertinence binaire (pour `precision@K`, `recall@K`, `MAP`) correspond au **grade 3**. Le `nDCG@10` utilise les grades comme gains `2^grade − 1`, donc un quasi-bon résultat (grade 1 ou 2) reçoit un crédit partiel. Après resserrement, le nombre de documents de grade 3 par requête est : **min 3, médiane 11,5, max 20, moyenne 11,75** (valeurs lues dans `outputs/summary.json`).

## Requêtes et styles

20 requêtes de test, chacune ciblant un triplet (topic, concept principal, terme de domaine) et étiquetée par un **style** :

- `lexical` : la requête reprend le vocabulaire exact du concept (avantage BM25) ;
- `semantic` : le concept est paraphrasé (avantage attendu des embeddings et de l'hybride) ;
- `mixed` : un mélange des deux.

Exemple paraphrasé (style `semantic`) :

```text
Q01 : retrieve documents that express the same idea
Concept attendu : semantic_search   Domaine : legal_tech
```

Exemple lexical (style `lexical`, cas de contrôle BM25) :

```text
Q03 : bm25 keyword matching term frequency scoring
Concept attendu : bm25   Domaine : recommender_systems
```

## Métriques

```text
precision@K = (pertinents de grade 3 dans le top K) / K
recall@K    = (pertinents de grade 3 dans le top K) / (total des pertinents de grade 3)
nDCG@K      = DCG@K / IDCG@K, avec gain = 2^grade - 1 et escompte 1/log2(rang+1)
MAP@N       = moyenne des average precision (pertinence binaire), classements tronqués à N
```

`K = 10`. La `MAP` est rapportée à profondeur `N = 50` (`candidate_count`) **pour tous les régimes**, afin que BM25, le dense et l'hybride — qui ne peut classer que ses 50 candidats — soient comparés à profondeur identique.

## Pipeline hybride et re-rankers

Étage 1 : BM25 (index inversé) renvoie les `N = 50` candidats. Étage 2 : un re-ranker réordonne ces candidats pour produire le top-`K`. Re-rankers comparés :

- **bi-encodeur** : cosinus entre l'embedding de la requête et ceux des candidats (déjà calculés) ;
- **cross-encodeur** : `cross-encoder/ms-marco-MiniLM-L-6-v2`, score conjoint de chaque paire (requête, document) ;
- **repli déterministe** : mélange cosinus + recouvrement lexical, clairement étiqueté `is_fallback=true`, jamais présenté comme un résultat cross-encodeur.

## Paramètres et environnement

```text
n_docs        = 2000          k (top-K)      = 10
n_queries     = 20            candidate_count= 50 (N, profondeur MAP)
random_state  = 42            embedder       = sentence-transformers/all-MiniLM-L6-v2 (384D)
HNSW          : M = 32, efSearch = 80, efConstruction par défaut Faiss
cross-encoder = cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers 5.6.0,
                revision c5ee24cb16019beea0893ab7796b1df96625c6b8)
```

Environnement des mesures (machine-dépendant pour les latences ; comparer les ratios, pas les valeurs absolues) :

```text
Python 3.13.3   NumPy 2.4.6   Faiss 1.14.3   Windows 11 (10.0.26200)   CPU AMD64
```

## Banc de passage à l'échelle

Vecteurs synthétiques en dimension 384, **clusterisés** (256 clusters, dispersion 0,08, graine de centroïdes fixe). Pour isoler l'effet de la taille, un **corpus maximal** est généré une fois et chaque taille (2k, 10k, 50k, 200k, 1M) est un **préfixe** de ce corpus ; le jeu de 20 requêtes est partagé. Sont mesurés : latence moyenne et écart-type (8 répétitions par requête), `recall@10` vs exact, temps de construction et empreinte mémoire (matrice brute vs index sérialisé). Paramètres et environnement sont écrits dans `outputs/scale_benchmark_meta.json`.

## Listings de code

Cœur du scoring BM25 sur index inversé (ne visite que les documents porteurs du terme) :

```python
for term in query_terms:
    posting_list = self.postings.get(term)   # [(doc_idx, tf), ...]
    if posting_list is None:
        continue
    idf = self.idf[term]
    for doc_idx, frequency in posting_list:
        doc_length = self.doc_lengths[doc_idx]
        denominator = frequency + self.k1 * (
            1.0 - self.b + self.b * doc_length / self.avg_doc_length
        )
        scores[doc_idx] += idf * frequency * (self.k1 + 1.0) / denominator
```

Cœur du pipeline hybride (BM25 -> re-ranking, le top-K est un ré-ordonnancement des candidats) :

```python
candidates = self.bm25.search(query_text, k=self.candidate_count)   # étage 1
rerank_scores = self.reranker.score(query_text, query_vector,
                                    candidate_indices, doc_texts, doc_vectors, bm25_scores)
order = np.argsort(-rerank_scores, kind="stable")                   # étage 2
results = [candidates[pos] for pos in order[:k]]
```

## Commandes de reproduction

Installation complète (vrais modèles) :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-full.txt
```

Expérience principale (BM25 / dense / hybride, métriques graduées) :

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --embedding-provider sentence-transformers --n-docs 2000 --output-dir outputs --k 10 --candidate-count 50
```

Banc de passage à l'échelle (jusqu'à 1M) et figures :

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_scale.py --full
.\.venv\Scripts\python.exe scripts\make_figures.py
```

Démonstration :

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Fichiers de sortie

- `outputs/evaluation.csv` : métriques par requête (BM25 vs dense) ;
- `outputs/regime_comparison.csv` et `outputs/regime_summary.json` : trois régimes + conclusion chiffrée ;
- `outputs/summary.json` : synthèse globale et métadonnées (modèles, repli, environnement) ;
- `outputs/scale_benchmark.csv` et `outputs/scale_benchmark_meta.json` : banc d'échelle + paramètres/environnement ;
- `outputs/figures/regime_comparison.png`, `scale_crossover.png`, `scale_recall_latency.png`, `distance_concentration.png`, `hnsw_layers.png` : figures du mémoire.
