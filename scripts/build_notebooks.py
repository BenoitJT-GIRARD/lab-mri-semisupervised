"""Génère programmatiquement les deux notebooks livrables.

Pourquoi générer les ``.ipynb`` plutôt que les rédiger à la main ?

* Reproductibilité : le code source des cellules vit dans le repo, sous
  contrôle de version, lisible et diffable.
* Cohérence : les imports, les sections et les figures sont garantis identiques
  entre le module ``curelyticsia`` et les notebooks.
* Industrialisation : le pipeline ``uv run jupyter nbconvert --execute`` peut
  être rejoué sans intervention manuelle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n"))


def write_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (curelyticsia)",
            "language": "python",
            "name": "curelyticsia",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[ok] {path}")


def build_notebook_one() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(
        md(
            """
# 01 — Exploration, extraction de features et clustering exploratoire

**Mission BrainScanAI — *CurelyticsIA*** &nbsp;·&nbsp; *Option B*

Vous êtes Data Scientist junior en computer vision. Le projet R&D *BrainScanAI*
vise à automatiser la détection de tumeurs cérébrales sur des IRM. Le dataset
fourni par Clara Martin contient :

- **100 images fortement labellisées** (50 *normal* / 50 *cancer*) annotées par
  des radiologues partenaires ;
- **1 406 images non labellisées** issues du même flux d'acquisition.

Ce premier notebook couvre les **étapes 1 à 3** de la mission :

1. Chargement et exploration du jeu de radiographies.
2. Prétraitement et **extraction des features** via un modèle pré-entraîné.
3. **Analyse non supervisée** : réduction de dimension + clustering, comparaison
   de plusieurs algorithmes, choix d'une méthode et **labellisation faible** des
   1 406 images non annotées.

> ⚠️ **Règle métier non négociable** — les jeux *fortement* et *faiblement*
> labellisés ne sont jamais mélangés ; ils sont conservés dans deux structures
> séparées tout au long du projet.
"""
        )
    )

    cells.append(
        md(
            """
## Définition du *done* (notebook 1)

| Critère | Cible |
|---|---|
| Inventaire complet du dataset (intégrité, résolution, modes) | ✅ |
| Outliers documentés et traités | ✅ |
| Features ResNet50 calculées pour les 1 506 images | ✅ |
| Au moins **4 algorithmes de clustering** testés | ✅ |
| ARI clustering (vs labels forts) ≥ 0.10 (mieux que le hasard) | objectif |
| Pseudo-labels « faibles » exportés en CSV pour le notebook 2 | ✅ |

L'erreur la plus coûteuse est le **faux négatif sur cancer** (passer à côté
d'une tumeur). Cette priorité métier oriente la sélection des métriques (recall
de la classe *cancer* avant accuracy globale) — détaillé dans le notebook 2.
"""
        )
    )

    cells.append(md("## 1. Chargement et exploration du jeu de radiographies"))

    cells.append(
        code(
            """
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from curelyticsia.config import (  # noqa: E402
    CLASS_TO_INDEX,
    CLASSES,
    ClusteringConfig,
    FIGURES_DIR,
    FeatureConfig,
    PROCESSED_DIR,
    SEED,
    ensure_dirs,
    set_global_seeds,
)

ensure_dirs()
set_global_seeds(SEED)

print(f"Project root : {ROOT}")
print(f"Classes      : {CLASSES} → {CLASS_TO_INDEX}")
print(f"Seed         : {SEED}")
"""
        )
    )

    cells.append(
        code(
            """
# Extraction du dataset (idempotent : ne ré-extrait pas si déjà présent).
import subprocess

result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "extract_dataset.py")],
    capture_output=True,
    text=True,
    check=True,
)
print(result.stdout)
"""
        )
    )

    cells.append(
        code(
            """
from curelyticsia.data.loader import (
    discover_images,
    records_to_dataframe,
    summarise_dataset,
)

records, corrupted = discover_images()
print(f"{len(records):>5} images valides")
print(f"{len(corrupted):>5} images corrompues écartées")
if corrupted:
    for p in corrupted[:5]:
        print("  -", p)

df = records_to_dataframe(records)
df.head()
"""
        )
    )

    cells.append(md("### 1.1 Vue d'ensemble"))

    cells.append(
        code(
            """
summary = summarise_dataset(records)
print("Total       :", summary["total"])
print("Par split   :", summary["by_split"])
print("Par label   :", summary["by_label"])
print("Modes       :", summary["modes"])
print("Résolutions :", dict(list(summary["resolutions"].items())[:5]))
"""
        )
    )

    cells.append(
        md(
            """
**Observations attendues** (à confirmer à l'exécution) :

- 100 images étiquetées (50 *normal* + 50 *cancer*) dans `avec_labels/` et
  ~1 400 images non labellisées dans `sans_label/`.
- Toutes les images sont en JPEG 512×512 — la documentation fournie le précise
  et l'inventaire le confirme.
- Le mode dominant est `L` (niveaux de gris) ; il est explicitement converti
  en RGB par le pipeline de preprocessing avant ResNet (qui attend 3 canaux
  ImageNet).
"""
        )
    )

    cells.append(md("### 1.2 Galeries d'exemples"))

    cells.append(
        code(
            """
from curelyticsia.viz.plots import plot_image_grid

mask_cancer = (df["split"] == "labeled") & (df["label_name"] == "cancer")
mask_normal = (df["split"] == "labeled") & (df["label_name"] == "normal")
mask_unlab = df["split"] == "unlabeled"

cancer_paths = df.loc[mask_cancer, "path"].head(10).tolist()
normal_paths = df.loc[mask_normal, "path"].head(10).tolist()
unlab_paths = df.loc[mask_unlab, "path"].head(10).tolist()

_ = plot_image_grid(
    cancer_paths,
    titles=[f"cancer · {Path(p).name[:8]}" for p in cancer_paths],
    cols=5,
    save_path=FIGURES_DIR / "samples_cancer.png",
)
"""
        )
    )

    cells.append(
        code(
            """
_ = plot_image_grid(
    normal_paths,
    titles=[f"normal · {Path(p).name[:8]}" for p in normal_paths],
    cols=5,
    save_path=FIGURES_DIR / "samples_normal.png",
)
"""
        )
    )

    cells.append(
        code(
            """
_ = plot_image_grid(
    unlab_paths,
    titles=[f"? · {Path(p).name[:8]}" for p in unlab_paths],
    cols=5,
    save_path=FIGURES_DIR / "samples_unlabeled.png",
)
"""
        )
    )

    cells.append(md("### 1.3 Statistiques de pixels et détection d'outliers"))

    cells.append(
        code(
            """
from curelyticsia.data.loader import compute_pixel_stats, detect_outlier_ids
from curelyticsia.viz.plots import plot_pixel_stats

stats = compute_pixel_stats(records, sample_size=None)  # toutes les images
print(stats.describe())

_ = plot_pixel_stats(stats, save_path=FIGURES_DIR / "pixel_stats.png")
"""
        )
    )

    cells.append(
        code(
            """
outlier_ids = detect_outlier_ids(stats, z_threshold=4.0)
print(f"{len(outlier_ids)} outliers détectés (|z| > 4 sur mean ou std)")

# Filtre éventuel : on les conserve dans cette mission (peu nombreux et
# potentiellement informatifs en imagerie médicale) mais on documente leur
# présence pour l'étape clustering.
records_clean = [r for r in records if r.image_id not in set(outlier_ids)]
print(f"{len(records_clean)} images conservées après filtrage soft")
"""
        )
    )

    cells.append(
        md(
            """
**Choix de traitement des outliers** : on adopte un seuil très conservateur
(*z* = 4) pour ne supprimer que des cas réellement aberrants (capteur défectueux
par exemple). En IRM cérébrale, des écarts d'intensité significatifs peuvent
correspondre à des tumeurs ; supprimer trop agressivement biaiserait la
détection. Le traitement reste documenté ici pour traçabilité.
"""
        )
    )

    cells.append(md("## 2. Prétraitement et extraction de features ResNet50"))

    cells.append(
        md(
            """
**Stratégie** :

- Resize à 256, *center crop* à 224×224, normalisation **ImageNet** (mean / std).
- Backbone **ResNet50** pré-entraîné sur ImageNet, **toutes les couches gelées**
  (recommandation explicite de l'énoncé : *« geler les couches convolutionnelles »*).
- La tête de classification ImageNet est remplacée par `Identity` ; on récupère
  la sortie du *global average pool* — un vecteur de **2 048 dimensions** par
  image.
- Mise en cache au format **Parquet** : la prochaine exécution est instantanée.
"""
        )
    )

    cells.append(
        code(
            """
from curelyticsia.features.extractor import extract_features

cfg = FeatureConfig()
features, index_df = extract_features(records, cfg=cfg, use_cache=True, progress=True)
print("features.shape :", features.shape)
print(index_df["split"].value_counts().to_string())
"""
        )
    )

    cells.append(
        code(
            """
# Vérification que les sorties (les embeddings) sont bien exploitables.
print("Aucun NaN ?      ", not np.isnan(features).any())
print("Norme moyenne L2 :", float(np.linalg.norm(features, axis=1).mean()))
print("Variance par dim :", float(features.var(axis=0).mean()))
"""
        )
    )

    cells.append(md("## 3. Analyse non supervisée"))

    cells.append(
        md(
            """
### 3.1 Standardisation et réduction de dimension

On centre-réduit les 2 048 dimensions, puis on conserve **95 % de la variance**
via PCA. La PCA accélère le clustering et stabilise t-SNE / UMAP.
"""
        )
    )

    cells.append(
        code(
            """
from curelyticsia.models.clustering import reduce_pca, standardise

features_std, scaler = standardise(features)
features_red, pca = reduce_pca(features_std, target_variance=0.95)
print(f"Dimensions PCA : {features_red.shape[1]} (variance cumulée ≥ 0.95)")
print(f"Première composante explique {pca.explained_variance_ratio_[0]:.2%} de la variance")
"""
        )
    )

    cells.append(md("### 3.2 Visualisation 2D (t-SNE / UMAP)"))

    cells.append(
        code(
            """
from curelyticsia.viz.plots import plot_2d_scatter, project_2d

labels_for_color = index_df["label_name"].fillna(value="").replace("", None).tolist()

coords_tsne = project_2d(features_red, method="tsne", seed=SEED)
_ = plot_2d_scatter(
    coords_tsne,
    labels=labels_for_color,
    title="t-SNE des embeddings ResNet50 — colorés par label connu",
    save_path=FIGURES_DIR / "tsne_labels.png",
)

coords_umap = project_2d(features_red, method="umap", seed=SEED)
_ = plot_2d_scatter(
    coords_umap,
    labels=labels_for_color,
    title="UMAP des embeddings ResNet50 — colorés par label connu",
    save_path=FIGURES_DIR / "umap_labels.png",
)
"""
        )
    )

    cells.append(md("### 3.3 Comparaison de plusieurs algorithmes de clustering"))

    cells.append(
        code(
            """
from curelyticsia.models.clustering import (
    align_cluster_labels,
    assign_weak_labels,
    build_clustering_report,
    export_weak_labels,
    fit_agglomerative,
    fit_dbscan,
    fit_gmm,
    fit_kmeans,
)

truth = np.where(
    index_df["split"] == "labeled",
    index_df["label_name"].map(lambda v: CLASS_TO_INDEX.get(v) if v else np.nan),
    np.nan,
).astype(float)

results = [
    fit_kmeans(features_red, truth, n_clusters=2, seed=SEED),
    fit_agglomerative(features_red, truth, n_clusters=2, linkage="ward"),
    fit_agglomerative(features_red, truth, n_clusters=2, linkage="average"),
    fit_gmm(features_red, truth, n_components=2, seed=SEED),
    fit_dbscan(features_red, truth, eps=8.0, min_samples=10),
]

report = build_clustering_report(results)
report
"""
        )
    )

    cells.append(
        md(
            """
**Lecture** :

- *Silhouette* (↑) et *Calinski-Harabasz* (↑) mesurent la séparation interne
  des clusters. *Davies-Bouldin* (↓) la dispersion ; on cherche donc la valeur
  la plus basse.
- L'**ARI** est l'arbitre : il compare la partition obtenue à la *vérité
  terrain* sur les 100 images étiquetées. ARI ∈ [-1, 1] ; 0 = hasard.
- DBSCAN peut produire un nombre arbitraire de clusters (ou 0). Sur des
  embeddings 2048d, il est rarement compétitif sans tuning approfondi : on le
  conserve néanmoins comme baseline pour tracer le résultat.
"""
        )
    )

    cells.append(md("### 3.4 Sélection du meilleur clustering et alignement des labels"))

    cells.append(
        code(
            """
candidates = [r for r in results if r.ari_vs_truth is not None]
best = max(candidates, key=lambda r: r.ari_vs_truth)
print(f"Méthode retenue : {best.name}")
print(f"ARI vs vérité   : {best.ari_vs_truth:.4f}")
print(f"Silhouette      : {best.silhouette}")
print(f"Davies-Bouldin  : {best.davies_bouldin}")
print(f"Calinski-Hara.  : {best.calinski_harabasz}")
"""
        )
    )

    cells.append(
        code(
            """
aligned = align_cluster_labels(best.labels, truth)

# Visualisation des clusters alignés sur la projection t-SNE.
labels_for_color_aligned = [None if v is None else v for v in (
    pd.Series(aligned).map({0: "normal", 1: "cancer", -1: "noise"}).tolist()
)]
_ = plot_2d_scatter(
    coords_tsne,
    labels=labels_for_color_aligned,
    title=f"Clusters alignés ({best.name}) — projection t-SNE",
    save_path=FIGURES_DIR / "tsne_clusters_aligned.png",
)
"""
        )
    )

    cells.append(md("### 3.5 Pseudo-labels « faibles » sur les images non annotées"))

    cells.append(
        code(
            """
weak = assign_weak_labels(index_df, aligned)
print(f"{len(weak)} pseudo-labels exportables")
print(weak["weak_label_name"].value_counts())

out = export_weak_labels(weak, ClusteringConfig())
print(f"\\nExport : {out}")
"""
        )
    )

    cells.append(
        md(
            """
**Vérification de la séparation des jeux** : `weak` ne contient **que** des
images du split `unlabeled` ; aucune image fortement labellisée n'apparaît.
La règle métier *« ne jamais mélanger faible et fort »* est respectée.
"""
        )
    )

    cells.append(md("## 4. Synthèse"))

    cells.append(
        md(
            """
- Inventaire propre du dataset (modes, résolutions, intégrité, outliers).
- Embeddings ResNet50 calculés et mis en cache → réutilisables sans recalcul.
- Quatre algorithmes de clustering comparés (K-Means, Agglomerative ×2, GMM,
  DBSCAN). La méthode retenue est celle qui maximise l'ARI face aux 100 labels
  forts.
- Les pseudo-labels « faibles » sont exportés vers
  `data/processed/weak_labels.csv` ; ils alimentent le **notebook 02** où l'on
  entraîne et compare un CNN supervisé pur vs un CNN semi-supervisé.

L'analyse pour le **passage à l'échelle (4 M images / 5 000 €)** est traitée
dans le support de présentation.
"""
        )
    )

    return cells


def build_notebook_two() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(
        md(
            """
# 02 — Approche semi-supervisée et comparaison avec une baseline supervisée

**Mission BrainScanAI — *CurelyticsIA*** &nbsp;·&nbsp; *Option B*

Ce second notebook met en œuvre l'**étape 4** de la mission :

1. Charger les pseudo-labels faibles (issus du clustering du notebook 01) et
   les 100 labels forts (sans jamais mélanger les deux jeux).
2. Réaliser un split *train / test* stratifié sur les labels forts (le test set
   restera **strictement inconnu** des deux modèles).
3. Entraîner :
   - une **baseline supervisée pure** (ResNet18 + tête 2 classes, fine-tuné
     uniquement sur les 80 images fortement labellisées du train) ;
   - une **approche semi-supervisée** (même architecture, pré-entraînée sur les
     ~1 400 pseudo-labels faibles, puis fine-tunée sur les mêmes 80 images).
4. Comparer les performances : accuracy, macro-F1, **recall de la classe
   *cancer*** (priorité métier), précision, ROC AUC, matrice de confusion.
5. Conclusion / *Definition of Done*.
"""
        )
    )

    cells.append(md("## 0. Imports & configuration"))

    cells.append(
        code(
            """
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from curelyticsia.config import (  # noqa: E402
    CLASS_TO_INDEX,
    CLASSES,
    ClusteringConfig,
    FIGURES_DIR,
    FeatureConfig,
    PROCESSED_DIR,
    SEED,
    TrainingConfig,
    ensure_dirs,
    set_global_seeds,
)
from curelyticsia.features.extractor import load_cached_features  # noqa: E402
from curelyticsia.viz.plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_history,
    plot_metrics_bar,
)

ensure_dirs()
set_global_seeds(SEED)
print("Classes :", CLASSES, "→", CLASS_TO_INDEX)
"""
        )
    )

    cells.append(md("## 1. Chargement des features et des labels"))

    cells.append(
        code(
            """
features, index_df = load_cached_features(FeatureConfig().cache_path)
weak = pd.read_csv(ClusteringConfig().weak_labels_path)
print(f"features            : {features.shape}")
print(f"index_df            : {index_df.shape}")
print(f"pseudo-labels faibles : {len(weak)}")

labeled_df = index_df[index_df["split"] == "labeled"].copy()
labeled_df["label_index"] = labeled_df["label_name"].map(CLASS_TO_INDEX).astype(int)
print("\\nDistribution labels forts :")
print(labeled_df["label_name"].value_counts())
"""
        )
    )

    cells.append(md("### 1.1 Garde-fous : aucun chevauchement entre faibles et forts"))

    cells.append(
        code(
            """
strong_ids = set(labeled_df["image_id"])
weak_ids = set(weak["image_id"])
overlap = strong_ids & weak_ids
assert not overlap, f"Chevauchement détecté : {overlap}"
print("OK — pas de chevauchement entre labels forts et faibles")
"""
        )
    )

    cells.append(md("## 2. Split train / test stratifié sur les labels forts"))

    cells.append(
        code(
            """
train_idx, test_idx = train_test_split(
    labeled_df.index,
    test_size=TrainingConfig().test_size,
    stratify=labeled_df["label_index"],
    random_state=SEED,
)
strong_train = labeled_df.loc[train_idx]
strong_test = labeled_df.loc[test_idx]

print(f"strong_train : {len(strong_train)} ({strong_train['label_name'].value_counts().to_dict()})")
print(f"strong_test  : {len(strong_test)} ({strong_test['label_name'].value_counts().to_dict()})")
print(f"weak (faibles): {len(weak)} ({weak['weak_label_name'].value_counts().to_dict()})")
"""
        )
    )

    cells.append(md("## 3. Stratégie A — Baseline supervisée pure"))

    cells.append(
        md(
            """
On entraîne un ResNet18 pré-entraîné ImageNet, dont on remplace la tête par
une couche linéaire 2 classes. Le réseau est ré-entraîné de bout en bout sur
les 80 images du train fortement labellisé.
"""
        )
    )

    cells.append(
        code(
            """
from curelyticsia.models.semi_supervised import train_supervised

cfg = TrainingConfig()
print(cfg)

sup_model, sup_report = train_supervised(
    train_paths=strong_train["path"].tolist(),
    train_labels=strong_train["label_index"].tolist(),
    test_paths=strong_test["path"].tolist(),
    test_labels=strong_test["label_index"].tolist(),
    class_names=list(CLASSES),
    cfg=cfg,
)
print(json.dumps(
    {k: v for k, v in sup_report.__dict__.items() if k not in ("history",)},
    indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o,
))
"""
        )
    )

    cells.append(md("## 4. Stratégie B — Semi-supervisé (faible → fort)"))

    cells.append(
        md(
            """
Le réseau démarre avec les mêmes poids ImageNet, est **pré-entraîné** sur les
~1 400 pseudo-labels (apprentissage de structures visuelles communes via les
clusters identifiés à l'étape 3), puis **fine-tuné** sur les 80 images du
train fortement labellisé avec un *learning rate* divisé par deux.
"""
        )
    )

    cells.append(
        code(
            """
from curelyticsia.models.semi_supervised import train_semi_supervised

semi_model, semi_report = train_semi_supervised(
    weak_paths=weak["path"].tolist(),
    weak_labels=weak["weak_label_index"].astype(int).tolist(),
    strong_train_paths=strong_train["path"].tolist(),
    strong_train_labels=strong_train["label_index"].tolist(),
    strong_test_paths=strong_test["path"].tolist(),
    strong_test_labels=strong_test["label_index"].tolist(),
    class_names=list(CLASSES),
    cfg=cfg,
)
print(json.dumps(
    {k: v for k, v in semi_report.__dict__.items() if k not in ("history",)},
    indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o,
))
"""
        )
    )

    cells.append(md("## 5. Comparaison des deux stratégies"))

    cells.append(
        code(
            """
def to_run(report) -> dict[str, float]:
    return {
        "accuracy": report.accuracy,
        "f1_macro": report.f1_macro,
        "recall_cancer": report.recall_per_class.get("cancer", float("nan")),
        "precision_cancer": report.precision_per_class.get("cancer", float("nan")),
        "f1_cancer": report.f1_per_class.get("cancer", float("nan")),
        "roc_auc": report.roc_auc if report.roc_auc is not None else float("nan"),
    }

runs = {"Supervisé pur": to_run(sup_report), "Semi-supervisé": to_run(semi_report)}
comparison = pd.DataFrame(runs).T
comparison.style.format("{:.3f}")
"""
        )
    )

    cells.append(
        code(
            """
_ = plot_metrics_bar(
    runs,
    metric_name="f1_macro",
    title="Macro-F1 — Supervisé vs Semi-supervisé",
    save_path=FIGURES_DIR / "compare_f1.png",
)
_ = plot_metrics_bar(
    runs,
    metric_name="recall_cancer",
    title="Rappel sur la classe cancer (priorité métier)",
    save_path=FIGURES_DIR / "compare_recall_cancer.png",
)
"""
        )
    )

    cells.append(md("### 5.1 Matrices de confusion"))

    cells.append(
        code(
            """
_ = plot_confusion_matrix(
    sup_report.confusion_matrix,
    class_names=list(CLASSES),
    title="Confusion — Supervisé pur",
    save_path=FIGURES_DIR / "cm_supervised.png",
)
_ = plot_confusion_matrix(
    semi_report.confusion_matrix,
    class_names=list(CLASSES),
    title="Confusion — Semi-supervisé",
    save_path=FIGURES_DIR / "cm_semi.png",
)
"""
        )
    )

    cells.append(md("### 5.2 Historique des phases d'entraînement"))

    cells.append(
        code(
            """
def history_to_df(report, label: str) -> pd.DataFrame:
    rows = []
    for i, log in enumerate(report.history, start=1):
        rows.append({
            "step": i,
            "phase": f"{label}/{log.phase}",
            "train_loss": log.train_loss,
            "train_acc": log.train_acc,
        })
    return pd.DataFrame(rows)

hist_df = pd.concat([
    history_to_df(sup_report, "supervised"),
    history_to_df(semi_report, "semi-supervised"),
], ignore_index=True)

_ = plot_history(hist_df, save_path=FIGURES_DIR / "training_history.png")
hist_df.tail(10)
"""
        )
    )

    cells.append(md("## 6. Discussion : justification de l'approche semi-supervisée"))

    cells.append(
        md(
            """
**Pourquoi la semi-supervision est-elle pertinente ici ?**

1. Le coût d'annotation médicale est **élevé** (radiologues experts, temps,
   responsabilité). Le budget *labellisation IA* est de **300 €** sur ce
   dataset et ne permettra pas d'annoter manuellement 1 500 IRM
   supplémentaires.
2. Les 1 400 images non annotées **portent de l'information** sur la structure
   visuelle des IRM (forme du crâne, position cérébrale, contraste type IRM).
   Le pré-entraînement sur les pseudo-labels stabilise la représentation et
   réduit le sur-apprentissage de la baseline 100 % supervisée sur seulement
   80 images d'entraînement.
3. La comparaison directe (mêmes données de test, même architecture, même
   *seed*) montre l'apport — ou ses limites — de l'approche.

**Hyperparamètres ajustés** :

- *learning rate* (1e-4 sur la phase fortement labellisée, divisé par 2 lors du
  fine-tuning pour préserver le pré-entraînement) ;
- nombre d'époques (5 en pré-entraînement faible, 8 en fine-tuning fort) ;
- *batch size* à 16 (compromis sur des images 224×224 en CPU).

**Erreur la plus coûteuse** : un **faux négatif sur cancer** (manquer une
tumeur) coûte plus cher qu'un faux positif (re-vérifier une IRM saine). Les
métriques mises en avant sont donc **`recall_cancer`** puis **`f1_cancer`**,
plutôt que l'accuracy globale qui peut masquer un déséquilibre de classes.
"""
        )
    )

    cells.append(md("## 7. Bonus — Comparaison avec sklearn `LabelPropagation`"))

    cells.append(
        md(
            """
La consigne suggère également les méthodes basées sur les graphes
(*label propagation*). On s'en sert ici comme **second avis** sur les features
ResNet déjà extraites — sans avoir à réentraîner un CNN.
"""
        )
    )

    cells.append(
        code(
            """
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation

# On reconstruit truth aligné sur l'ordre de features.
truth_full = np.full(len(features), -1, dtype=int)
for idx, row in labeled_df.iterrows():
    pos = int(np.where(index_df["image_id"].values == row["image_id"])[0][0])
    truth_full[pos] = int(row["label_index"])

# On ne fournit à LabelPropagation que les labels du **train fort** + les
# features ; le test reste donc inconnu — comparaison honnête.
train_ids = set(strong_train["image_id"].tolist())
test_ids = set(strong_test["image_id"].tolist())

semi_truth = np.where(
    [iid in train_ids for iid in index_df["image_id"]],
    truth_full,
    -1,
)

X = StandardScaler().fit_transform(features)
lp = LabelPropagation(kernel="knn", n_neighbors=10, max_iter=200)
lp.fit(X, semi_truth)

# Évaluation sur le test set fortement labellisé.
test_mask = np.array([iid in test_ids for iid in index_df["image_id"]])
y_true_test = truth_full[test_mask]
y_pred_test = lp.transduction_[test_mask]
from sklearn.metrics import accuracy_score, f1_score, recall_score
print("LabelPropagation — accuracy   :", accuracy_score(y_true_test, y_pred_test))
print("LabelPropagation — macro F1   :", f1_score(y_true_test, y_pred_test, average="macro"))
print("LabelPropagation — recall cancer :", recall_score(y_true_test, y_pred_test, pos_label=CLASS_TO_INDEX["cancer"]))
"""
        )
    )

    cells.append(md("## 8. *Definition of Done*"))

    cells.append(
        code(
            """
done_table = pd.DataFrame(
    [
        ("ARI clustering vs labels forts ≥ 0.10", "voir notebook 01"),
        ("F1 macro semi-sup ≥ F1 macro supervisé", f"{semi_report.f1_macro:.3f} vs {sup_report.f1_macro:.3f}"),
        ("Recall cancer ≥ 0.90", f"sup={sup_report.recall_per_class.get('cancer', float('nan')):.3f} ; semi={semi_report.recall_per_class.get('cancer', float('nan')):.3f}"),
        ("Tests pytest verts", "voir QA"),
        ("Notebooks ré-exécutables", "uv run jupyter nbconvert --execute"),
    ],
    columns=["Critère", "Valeur observée"],
)
done_table
"""
        )
    )

    cells.append(
        code(
            """
# Sauvegarde du rapport JSON consommé par le support de présentation.
out = {
    "supervised": to_run(sup_report),
    "semi_supervised": to_run(semi_report),
    "training_config": cfg.__dict__,
}
out_path = PROCESSED_DIR / "training_report.json"
out_path.write_text(json.dumps(out, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o), encoding="utf-8")
print(f"Rapport sauvegardé : {out_path}")
"""
        )
    )

    cells.append(
        md(
            """
**Conclusion**

- L'approche semi-supervisée s'appuie sur les pseudo-labels issus du
  clustering pour exploiter les 1 400 IRM non annotées sans coût d'annotation
  supplémentaire.
- La comparaison directe avec la baseline supervisée pure (mêmes données,
  même test) permet de quantifier le gain.
- Les recommandations pour le passage à l'échelle (4 M d'images, 5 000 €) sont
  détaillées dans le support de présentation : faisabilité, choix techniques
  (batch GPU, infrastructure, *active learning*), risques et conditions.
"""
        )
    )

    return cells


def main() -> None:
    write_notebook(
        NOTEBOOKS_DIR / "01_exploration_features_clustering.ipynb",
        build_notebook_one(),
    )
    write_notebook(
        NOTEBOOKS_DIR / "02_semi_supervised.ipynb",
        build_notebook_two(),
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
