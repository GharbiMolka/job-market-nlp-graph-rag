# Analyse intelligente des offres d'emploi Data — NLP, graphes et Graph-RAG

Projet de data science qui exploite des offres d'emploi du domaine Data pour analyser le marché, classifier les postes, construire un **graphe hétérogène Entreprises – Postes – Compétences**, expérimenter un **GNN**, recommander des compétences et répondre à des questions à partir du graphe (**Graph-RAG simplifié**).

Tout le travail est réalisé dans un notebook unique, exécutable sur Google Colab ou en local.

---

## Objectifs

- Analyser les tendances du marché : compétences les plus demandées, pays, évolution dans le temps, salaires (lorsque ces champs existent).
- Classifier les offres par catégorie de poste (`job_title_short`) à partir du titre et des compétences.
- Représenter les titres et compétences avec **TF-IDF**, **Word2Vec** et **SBERT**.
- Regrouper les postes et les compétences par **clustering** (K-Means).
- Extraire et normaliser les entités (entreprises, compétences).
- Construire un graphe hétérogène et détecter des **communautés** (Louvain).
- Prédire des liens Poste–Compétence avec un **GNN de type GraphSAGE** (PyTorch natif).
- **Recommander des compétences** et évaluer la recommandation (Precision@K / Recall@K).
- Répondre à des questions en langage naturel via un **mini Graph-RAG**.
- Exposer le tout dans une application **Streamlit**.

## Dataset

[`lukebarousse/data_jobs`](https://huggingface.co/datasets/lukebarousse/data_jobs) (Hugging Face), chargé automatiquement via la bibliothèque `datasets`.

> **Limite importante.** Ce dataset fournit des champs structurés (titre, entreprise, pays, compétences, dates, salaires selon les lignes) mais pas nécessairement de description libre. Sans description, le module d'extraction exploite `company_name`, `job_title` et `job_skills` : il s'agit d'une **normalisation d'entités à partir de champs structurés, pas d'un NER complet sur texte libre**. Si une colonne de description est détectée, le notebook bascule automatiquement sur un NER spaCy + PhraseMatcher.

## Contenu du notebook

| Section | Contenu |
|---|---|
| 0 | Installation et configuration (paramètres modifiables) |
| 1 | Chargement, exploration, nettoyage et normalisation |
| 2 | Tendances du marché (compétences, géographie, temps, salaires) |
| 3 | NLP : prétraitement spaCy, TF-IDF, SBERT, Word2Vec |
| 4 | Classification des postes, avec séparation train/test sans fuite d'information |
| 5 | Clustering des postes et des compétences |
| 6 | Extraction d'entités (NER adaptatif) et enrichissement sémantique |
| 7 | Graphe hétérogène Entreprises–Postes–Compétences, centralités, Louvain |
| 8 | GNN GraphSAGE simplifié : prédiction de liens Poste–Compétence |
| 9 | Recommandation de compétences et évaluation |
| 10 | Mini Q&A basé sur le graphe (Graph-RAG), génération LLM optionnelle (Ollama) |
| 11 | Export des modèles et résultats dans `artifacts/` |
| 12 | Génération de l'application Streamlit `app.py` |
| 13 | Bilan et limites |

## Installation

Python 3.10 ou plus récent est recommandé.

```bash
git clone https://github.com/<votre-utilisateur>/<nom-du-depot>.git
cd <nom-du-depot>

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Utilisation

### 1. Exécuter le notebook

```bash
jupyter notebook notebooks/job_market_nlp_graph.ipynb
```

Exécuter toutes les cellules dans l'ordre. Le premier lancement télécharge le dataset et le modèle SBERT (`all-MiniLM-L6-v2`).

Paramètres modifiables dans la cellule de configuration :

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| `SAMPLE_SIZE` | `20000` | Taille de l'échantillon (`None` = toutes les lignes, plus de RAM et de temps) |
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | Modèle d'embeddings |
| `N_COMPANIES_GRAPH` | `150` | Nombre d'entreprises dans le graphe |
| `N_SKILLS_GRAPH` | `60` | Nombre de compétences dans le graphe |
| `MIN_EDGE_SUPPORT` | `3` | Support minimal d'une arête |

### 2. Lancer l'application Streamlit

La cellule 12 du notebook génère `app.py` et la cellule 11 produit le dossier `artifacts/`. Une fois le notebook exécuté :

```bash
streamlit run app.py
```

### 3. Google Colab

Ouvrir le notebook dans Colab et exécuter la cellule d'installation en tête de notebook.

## Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── job_market_nlp_graph.ipynb
├── artifacts/        # généré par le notebook (modèles, graphe, métriques)
└── app.py            # généré par le notebook (interface Streamlit)
```

## Méthodologie et choix techniques

- **Pas de fuite d'information** : séparation train/test (`GroupShuffleSplit` sur les titres) avant l'ajustement de TF-IDF et de Word2Vec ; SBERT est un encodeur pré-entraîné figé.
- **Graphe tripartite** : les arêtes Entreprise–Poste et Poste–Compétence portent le support (nombre d'offres) et une association PMI positive.
- **GNN en PyTorch natif** pour éviter l'installation fragile de PyTorch Geometric.
- **Graph-RAG traçable** : la réponse est construite de façon déterministe à partir des faits récupérés dans l'index et le graphe ; un LLM local (Ollama) peut ensuite la reformuler, sans inventer d'entreprises ni de chiffres.

## Limites

1. Sans description libre dans le dataset, le NER complet sur texte d'offre n'est pas réalisable.
2. Le GNN est une expérimentation GraphSAGE simplifiée ; ce n'est pas un GTN (Graph Transformer Network).
3. Les résultats dépendent de l'échantillon, des seuils de construction du graphe et de la distribution du dataset.
4. Les recommandations et le Q&A ne sont pas des prédictions garanties du marché réel.
5. Les salaires et les dates ne sont analysés que si les colonnes existent et contiennent assez de valeurs valides.

## Technologies

Python · pandas · NumPy · scikit-learn · spaCy · Sentence-Transformers · Gensim · NetworkX · PyTorch · Hugging Face Datasets · Matplotlib / Seaborn · Streamlit

## Auteur

<Votre nom> — <lien LinkedIn ou email>

## Licence

À définir (par exemple MIT). Le dataset `lukebarousse/data_jobs` reste soumis à sa propre licence.
