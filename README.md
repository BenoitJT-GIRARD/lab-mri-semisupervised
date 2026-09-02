# BrainScanAI — exploration semi-supervisée d'IRM cérébrales

Projet R&D *CurelyticsIA*. Mission Option B : explorer un jeu de 1 506 images IRM
(100 labellisées normal/cancer, 1 406 non labellisées), extraire des features visuelles via un
modèle pré-entraîné, appliquer du clustering exploratoire puis une approche semi-supervisée
(CNN pré-entraîné sur labels faibles puis fine-tuné sur labels forts).

## Structure

```
CurelyticsIA/
├── data/
│   ├── raw/                 # dataset extrait (ignoré par git)
│   ├── interim/
│   └── processed/           # features.parquet, weak_labels.csv (ignoré)
├── src/mri_semisupervised/        # package Python
│   ├── config.py            # paths, seeds, hyperparamètres
│   ├── data/loader.py       # discovery, intégrité, métadonnées
│   ├── data/preprocess.py   # transforms ImageNet
│   ├── features/extractor.py# backbone ResNet50 → embeddings 2048d
│   ├── models/clustering.py # K-Means / DBSCAN / Agglomerative / GMM + ARI
│   ├── models/semi_supervised.py  # CNN + boucles train/eval
│   └── viz/plots.py         # grilles d'images, t-SNE, ROC, CM
├── tests/                   # pytest, fixtures synthétiques
├── notebooks/
│   ├── 01_exploration_features_clustering.ipynb
│   └── 02_semi_supervised.ipynb
├── reports/
│   ├── figures/             # PNG embarquées dans la présentation
│   ├── presentation.pdf     # 15 slides
│   └── auto_evaluation.pdf  # grille FAE remplie
├── scripts/
│   ├── extract_dataset.py
│   ├── build_features.py
│   ├── run_clustering.py
│   └── train_semi_supervised.py
├── pyproject.toml           # uv + ruff + pytest + bandit
└── README.md
```

## Setup (uv)

```powershell
uv sync --extra dev
uv run python -m ipykernel install --user --name mri_semisupervised --display-name "Python (mri_semisupervised)"
```

## Pipeline reproductible

```powershell
# 1. Extraction du dataset depuis infos/ (idempotent)
uv run python scripts/extract_dataset.py

# 2. Extraction des features ResNet50 (cache parquet)
uv run python scripts/build_features.py

# 3. Clustering exploratoire (génère weak_labels.csv + figures)
uv run python scripts/run_clustering.py

# 4. Entraînement semi-supervisé + comparaison supervisée
uv run python scripts/train_semi_supervised.py

# 5. Notebooks (re-exécution end-to-end)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploration_features_clustering.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_semi_supervised.ipynb
```

## Qualité

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run bandit -r src/ -q
```

## Livrables (convention OpenClassrooms)

Le dossier `reports/Bijeyti_Ben_Notebook_052026/` contient les livrables nommés selon
la convention `Nom_Prenom_n_nom_mmaaaa`. Si votre identité diffère, exécutez :

```powershell
uv run python scripts/package_deliverables.py --nom <Nom> --prenom <Prenom>
```

## Définition du *done*

- ARI clustering K-Means vs labels forts ≥ 0.10 (au-dessus du hasard).
- F1 semi-supervisé ≥ F1 supervisé pur sur jeu de test.
- **Recall classe « cancer » ≥ 0.90** (priorité métier : le faux négatif est l'erreur la
  plus coûteuse en e-santé).
- Tests pytest verts, ruff/bandit propres.
- Notebooks ré-exécutables end-to-end depuis `uv run jupyter nbconvert --execute`.

## Contraintes métier

- Budget actuel labellisation IA : **300 €** sur le présent dataset.
- Cible scaling : **4 000 000 d'images** à labelliser pour **5 000 €** (≈ 0,00125 €/image).
  Le passage à l'échelle est étudié dans la présentation (slides 12-13).

## Licence

MIT — usage académique.
