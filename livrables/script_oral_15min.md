# Script oral - 10 a 15 minutes

Les slides sont autoportantes : elles contiennent tous les points a aborder. Ce script ne sert qu'a derouler le raisonnement a l'oral, pas a etre lu. Fil : bases -> probleme -> methodes -> projet -> resultats -> demo -> conclusion.

## 0:00 - 0:40 | Slide 1 - Titre

Bonjour, nous presentons le sujet 9 : bases vectorielles et recherche semantique. L'idee : retrouver les bons documents meme quand l'utilisateur n'emploie pas les memes mots que les documents. Deux questions nous guident : (1) l'hybride BM25 + re-ranking bat-il chaque approche seule, et (2) a partir de quelle taille HNSW devient-il indispensable.

## 0:40 - 1:25 | Slide 2 - Requete, document, resultat

Trois mots a distinguer. La requete : ce que l'utilisateur tape. Le document : un texte deja stocke. Le resultat : un document que le moteur affiche apres avoir compare la requete a la base. Phrase a retenir : tous les resultats sont des documents, mais tous les documents ne deviennent pas des resultats.

## 1:25 - 2:15 | Slide 3 - Le probleme

Une requete et un document peuvent parler de la meme chose sans les memes mots. Exemple : "reperer un paiement anormal" vs "detection de transactions frauduleuses". La recherche lexicale cherche les mots communs et risque de passer a cote ; la recherche semantique rapproche par le sens. Enjeu RAG : si la recherche rate le bon passage, la reponse generee est fausse.

## 2:15 - 3:15 | Slide 4 - BM25

Premiere brique : BM25, methode lexicale. Index inverse : chaque mot pointe vers les documents qui le contiennent. L'IDF donne plus de poids aux mots rares (HNSW vaut plus que "le"). k1 gere la repetition, b la longueur des documents. Point a souligner : on l'a implemente nous-memes, from scratch, donc la formule est entierement explicable. Force : tres bon sur les mots exacts ; limite : aveugle aux synonymes.

## 3:15 - 4:15 | Slide 5 - Embeddings

Deuxieme brique : les embeddings. On transforme le texte en vecteur dense ; le modele all-MiniLM-L6-v2 produit 384 dimensions. Requete et documents dans le meme espace, compares par similarite cosinus (l'angle ; apres normalisation, un produit scalaire). Sens proche = vecteurs proches. La heatmap des 20 phrases rend l'idee visible : meme theme, similarite plus elevee.

## 4:15 - 5:10 | Slide 6 - Recherche exacte et echelle

Une fois les vecteurs obtenus, il faut trouver les plus proches. La methode exacte compare a tous les documents : fiable mais en O(n*d), donc lourde quand le corpus grandit. En haute dimension, les distances se concentrent (coefficient de variation 0,505 en dim 2 -> 0,023 en dim 768), et les index exacts eliminent moins de regions. D'ou les methodes approchees.

## 5:10 - 6:10 | Slide 7 - HNSW

HNSW : une approximation controlee pour aller bien plus vite. Graphe multi-couches : les couches hautes servent de raccourcis, la couche basse contient tous les vecteurs. La recherche est un BFS guide par la distance : on explore les voisins prometteurs sans parcourir tout le graphe. Comme c'est approximatif, on mesure la qualite par le recall@K face a l'exact.

## 6:10 - 7:00 | Slide 8 - Notre projet

On a construit un banc d'essai reproductible (graine 42) : 2 000 documents synthetiques et 20 requetes annotees. L'interet du synthetique : on connait la verite terrain, donc des metriques fiables. On compare quatre regimes : BM25, dense, hybride, et HNSW pour l'indexation.

## 7:00 - 7:55 | Slide 9 - Pipeline hybride

L'hybride en deux etapes. D'abord BM25 recupere 50 candidats (etape de rappel). Ensuite un re-ranker reordonne ces 50 pour produire le top 10 (cross-encoder ms-marco-MiniLM). Point cle : le re-ranker ne voit que ces 50 candidats. Si BM25 rate un bon document, l'hybride ne peut pas le rattraper. Cette contrainte explique nos resultats.

## 7:55 - 8:45 | Slide 10 - Evaluation

Avant de comparer, il fallait une mesure correcte. Une pertinence binaire etait trop large : trop de documents pertinents, donc des scores satures. On a utilise une pertinence graduee 0 a 3 (grade 3 = bon theme + bon concept principal + bon domaine). On mesure precision@10, recall@10, nDCG@10 et MAP@50. Le nDCG compte l'ordre : un bon document en premier vaut mieux qu'en dixieme.

## 8:45 - 10:20 | Slide 11 - Resultat hybride

Premier resultat, nuance. En nDCG@10 : BM25 0,540, dense 0,617, hybride 0,614. L'hybride bat nettement BM25, mais fait jeu egal avec le dense (0,003 d'ecart, c'est du bruit). Pourquoi : il est plafonne par le premier etage BM25. En revanche, sur les requetes semantiques, l'hybride depasse les deux : 0,508 contre 0,478 pour le dense et 0,377 pour BM25. Sa valeur n'est donc pas un gain partout, mais une robustesse ciblee quand lexical et semantique se completent.

## 10:20 - 11:45 | Slide 12 - Resultat HNSW

Deuxieme resultat. A petite taille, HNSW n'apporte rien (l'exact est deja instantane). La bascule pratique arrive vers 50 000 vecteurs : HNSW devient nettement plus rapide. A 1 million, il est environ 48 fois plus rapide que la force brute. Le cout : le recall baisse a efSearch fixe (jusqu'a 0,75 a 1M) et l'index pese 18 % de plus en memoire. On recupere du recall en augmentant efSearch, au prix de la latence.

## 11:45 - 13:00 | Slide 13 - Demo Streamlit

La demo rend les resultats visibles. Je montre d'abord une requete semantique avec BM25, dense et hybride cote a cote : BM25 remonte les documents proches par les mots, le dense et l'hybride retrouvent des formulations proches par le sens. Puis le re-ranking : quels documents le cross-encoder fait monter ou descendre dans le top 10. Enfin je bouge efSearch : le recall remonte mais la latence augmente. Si la demo ne se lance pas, les courbes des slides servent de plan B.

## 13:00 - 14:15 | Slide 14 - Conclusion

Notre projet ne montre pas qu'une methode gagne toujours, mais que chacune a son domaine : BM25 pour les mots exacts, les embeddings pour le sens, l'hybride sur les requetes semantiques, HNSW pour le passage a l'echelle. Limite principale : le corpus est synthetique, donc on defend des tendances, pas des magnitudes absolues. Suites possibles : fusionner BM25 et dense avant re-ranking, ou un deploiement avec pgvector et des filtres SQL.

Phrase finale possible : notre apport n'est pas de dire qu'une methode remplace les autres, mais de montrer dans quels cas chacune devient pertinente.
