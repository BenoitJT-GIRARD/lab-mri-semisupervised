"""Generate the two notebooks programmatically.

Why build the ``.ipynb`` rather than write them by hand:

* the source of every cell lives in the repository, versioned, readable and diffable;
* the imports and the sections cannot drift away from the ``mri_semisupervised`` package,
  because they are written against it here;
* ``uv run python -m nbconvert --execute`` replays them with no manual step, so the stored
  outputs are always a capture of a real run rather than an edited one.

Usage:
    uv run python scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"

PREAMBLE = """
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
"""


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n"))


def write_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python (mri_semisupervised)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(f"[ok] {path}")


# --------------------------------------------------------------------------- #
# 01 — the dataset
# --------------------------------------------------------------------------- #
def build_dataset_notebook() -> list[nbf.NotebookNode]:
    return [
        md("""
# 01 — The dataset, and what hashing it revealed

1 506 MRI files: a labelled pool of 100 (normal / cancer) and an unlabelled pool of 1 406.
The plan was to embed them with a frozen ResNet50, cluster the embeddings, and turn the
clusters into pseudo-labels for a semi-supervised model.

This notebook starts one step earlier, with a question the original pipeline never asked:
**are these 1 506 files 1 506 images?**
"""),
        code(PREAMBLE),
        md("""
## 1. Identity comes from content, not from the path

The first version of this pipeline identified an image by an MD5 of its **file path**. Two
copies of one scan filed in two folders therefore received two different identifiers — and
the guard meant to keep the labelled images out of the unlabelled pool, an exclusion on the
folder name, had nothing to catch them with.

Here an image is what it contains.
"""),
        code("""
from mri_semisupervised.config import LABELED_DIR, UNLABELED_DIR
from mri_semisupervised.data.manifest import apply_duplicate_rules, build_manifest, summarise

manifest = apply_duplicate_rules(build_manifest(LABELED_DIR, UNLABELED_DIR))
facts = summarise(manifest)
for key, value in facts.items():
    print(f"{key:34} {value}")
"""),
        md("""
## 2. What the hashes found

Three findings, none of them visible from the folder listing:

* the labelled pool holds **one image twice**, so the evaluation set is 99 images, not 100 —
  and the class balance is 50 / 49, not the announced 50 / 50;
* **31 of those 99 evaluation images also sit in the unlabelled pool**, byte for byte. The
  semi-supervised arm pre-trains on that pool, so a third of every test fold was already
  seen — whatever the cross-validation does;
* the unlabelled pool carries 63 redundant copies, which silently weight the pre-training.

The leak is also **asymmetric**: 23 `normal` against 8 `cancer`. It does not add noise, it
leans.
"""),
        code("""
leaked_ids = set(
    manifest.loc[
        (manifest["pool"] == "unlabelled") & (manifest["exclusion_reason"] == "copy_of_labelled"),
        "image_id",
    ]
)
leaked = manifest[(manifest["pool"] == "labelled") & manifest["image_id"].isin(leaked_ids)]
print(f"evaluation images with a copy in the unlabelled pool: {leaked['image_id'].nunique()}")
leaked.drop_duplicates("image_id")["label"].value_counts().rename("images").to_frame()
"""),
        md("""
### The rules, written down because they are decisions

| situation | decision | why |
|---|---|---|
| labelled image duplicated in the unlabelled pool | leaves **that pool** | removing it from the evaluation would be choosing the test set after seeing it |
| redundant copies inside the unlabelled pool | one kept | otherwise they weight the pre-training without saying so |
| an image repeated inside the labelled pool | one kept | kept twice, it sits on both sides of a fold |
| the same content under two labels | **build stops** | that is not a duplicate, it is a contradiction |
"""),
        code("""
manifest["exclusion_reason"].fillna("kept").value_counts().rename("files").to_frame()
"""),
        md("""
## 3. What the images look like

Everything is 512x512 RGB. The two classes differ in intensity distribution, which is worth
knowing before reading anything into a clustering that separates them.
"""),
        code("""
from mri_semisupervised.data.loader import compute_pixel_stats, discover_images
from mri_semisupervised.viz.plots import plot_image_grid, plot_pixel_stats

records, corrupted = discover_images()
print(f"{len(records)} readable files, {len(corrupted)} unreadable")

stats = compute_pixel_stats(records, sample_size=300, seed=0)
plot_pixel_stats(stats)
"""),
        code("""
labelled = manifest[(manifest["pool"] == "labelled") & manifest["kept_for_training"]]
sample = labelled.groupby("label").head(4)
plot_image_grid(sample["path"].tolist(), titles=sample["label"].tolist(), cols=4)
"""),
        md("""
### Histogram equalisation

Stretching the intensities makes structure easier to see. It is used here for looking, not
for the features: the backbone was trained under ImageNet normalisation, and feeding it
something else would trade a small visual gain for a distribution shift.
"""),
        code("""
from mri_semisupervised.viz.plots import plot_equalization_demo

plot_equalization_demo(sample["path"].iloc[0])
"""),
        md("""
## 4. Clustering the embeddings

Five algorithms on the ResNet50 embeddings, compared on internal metrics and on the ARI
against the labels.

**Where the ARI is computed matters.** In the corrected protocol it is computed on the
*training fold's* labels only, inside the fold. Here, in exploration, it is computed on all
of them — which is legitimate for looking at the data, and was exactly the mistake when the
same number was used to *choose* the method that produced the pseudo-labels.
"""),
        code("""
from mri_semisupervised.config import ClusteringConfig, FeatureConfig
from mri_semisupervised.features.extractor import load_cached_features
from mri_semisupervised.models.clustering import (
    build_clustering_report,
    fit_agglomerative,
    fit_dbscan,
    fit_gmm,
    fit_kmeans,
    reduce_pca,
    standardise,
)

features, index_df = load_cached_features(FeatureConfig().cache_path)
truth = index_df["label_index"].to_numpy(dtype=float)

reduced, _ = standardise(features)
reduced, _ = reduce_pca(reduced, target_variance=ClusteringConfig().pca_variance)
print(f"{features.shape[1]} dimensions -> {reduced.shape[1]} components at 95% variance")

results = [
    fit_kmeans(reduced, truth, n_clusters=2),
    fit_agglomerative(reduced, truth, n_clusters=2, linkage="ward"),
    fit_agglomerative(reduced, truth, n_clusters=2, linkage="average"),
    fit_gmm(reduced, truth, n_components=2),
    fit_dbscan(reduced, truth, eps=8.0, min_samples=10),
]
build_clustering_report(results)
"""),
        md("""
### What the table says, and what it does not

Read the silhouette column beside the ARI column. `Agglomerative(average)` has the best
silhouette of the five and an ARI of zero: it found a very tight structure that has nothing
to do with the diagnosis — one cluster holding almost everything, and one holding a handful
of outliers.

An internal metric measures whether a partition is *neat*. It cannot tell you whether it is
*the one you wanted*. That is the whole reason the ARI is in the table.

DBSCAN, at this eps, declares nearly everything noise. Reported rather than tuned away: a
method that refuses to split is information about the embedding space.
"""),
        code("""
from mri_semisupervised.viz.plots import plot_2d_scatter, project_2d

best = max((r for r in results if r.ari_vs_truth is not None), key=lambda r: r.ari_vs_truth)
print(f"best ARI against the labels: {best.name} ({best.ari_vs_truth:.3f})")

coords = project_2d(reduced, method="tsne", seed=42)
labels = index_df["label_name"].fillna("unlabeled").tolist()
plot_2d_scatter(coords, labels, title="t-SNE of the ResNet50 embeddings, coloured by label")
"""),
        md("""
## 5. Where this leaves the pseudo-labels

The best clustering reaches an ARI around 0.6 against the true labels. That is substantial
and far from decisive: the pseudo-labels it produces will be right often enough to be worth
trying, and wrong often enough that the trying has to be measured rather than assumed.

Notebook 02 measures it — under a protocol where the test fold takes part in no decision,
and against a control that receives the same images with their pseudo-labels shuffled.
"""),
    ]


# --------------------------------------------------------------------------- #
# 02 — the protocol
# --------------------------------------------------------------------------- #
def build_protocol_notebook() -> list[nbf.NotebookNode]:
    return [
        md("""
# 02 — The protocol, and what it says

The first version of this comparison concluded that the semi-supervised arm improved recall
on the cancer class, 0.900 to 0.960. Three leaks stood behind that number, all of them
pushing the same way:

1. **31 of the 99 evaluation images were in the pre-training pool** (notebook 01);
2. the **clustering method** was chosen by ARI against every label, test folds included;
3. the **cluster-to-class alignment** was decided by a vote over those same labels.

And a confound: the semi-supervised arm took strictly more gradient steps than its baseline.

This notebook reads the artefacts of the corrected protocol. It is not there to defend a
conclusion — it is there to report one.
"""),
        code(PREAMBLE),
        md("""
## 1. What the corrected protocol does differently

* Duplicates are removed **by content**, and the evaluation set never loses an image.
* The clustering, the choice of method and the alignment happen **inside the training
  fold**. The test fold takes part in no decision.
* Three arms share folds, architecture and starting weights:

| arm | pre-training | what it isolates |
|---|---|---|
| `supervised` | none | the reference |
| `semi_supervised` | the fold's pseudo-labels | the supposed contribution |
| `permuted_control` | the same images, labels shuffled | budget and exposure |

* The checkpoint and the decision threshold come from an inner validation split carved out
  of the training fold.
* Five repeats of a five-fold cross-validation, so the spread is measured rather than
  guessed.

The third arm is the one that can end the discussion. It sees the same images and takes the
same steps; only the pairing between image and pseudo-label is destroyed.
"""),
        code("""
import json

from mri_semisupervised.config import EXPERIMENTS_DIR

runs = sorted(p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
corrected = [p for p in runs if p.name.startswith("corrected")][-1]
print(f"reading {corrected.name}")

meta = json.loads((corrected / "manifest.json").read_text(encoding="utf-8"))
print(f"dataset fingerprint : {meta['dataset_fingerprint']}")
print(f"evaluation images   : {meta['evaluation_images']}")
print(f"unlabelled pool     : {meta['unlabelled_pool']}")
print(f"folds               : {meta['protocol']['n_splits']} x {meta['protocol']['n_repeats']} repeats")
print(f"gpu                 : {meta['versions']['gpu']}")

per_fold = pd.read_parquet(corrected / "per_fold.parquet")
predictions = pd.read_parquet(corrected / "predictions.parquet")
folds = pd.read_parquet(corrected / "folds.parquet")
"""),
        md("""
## 2. The three arms, side by side

Two families of number, and they answer different questions. **ROC AUC and PR-AUC** say how
well the scores rank, whatever threshold is applied. **Recall and F1** say what happens at
the threshold that was actually chosen — on the inner validation, never on the test fold.

The original comparison reported only the second kind, which is how a model whose ranking
had got *worse* came to look better.
"""),
        code("""
headline = ["roc_auc", "pr_auc", "recall_positive", "f1_macro", "accuracy"]
per_fold.groupby("arm")[headline].agg(["mean", "std"]).round(3)
"""),
        md("""
## 3. Paired comparisons

The arms share their folds, so the comparison is paired. Treating the two as independent
samples would throw the pairing away and widen every interval for nothing.
"""),
        code("""
from mri_semisupervised.protocol.uncertainty import paired_difference

pivot = per_fold.pivot(index="fold", columns="arm")
rows = []
for metric in ["roc_auc", "pr_auc", "recall_positive"]:
    for a, b in [
        ("semi_supervised", "supervised"),
        ("semi_supervised", "permuted_control"),
        ("permuted_control", "supervised"),
    ]:
        out = paired_difference(pivot[(metric, a)].to_numpy(), pivot[(metric, b)].to_numpy())
        rows.append({"metric": metric, "comparison": f"{a} - {b}", **out})
pd.DataFrame(rows).round(4)
"""),
        md("""
The line to read first is `semi_supervised - permuted_control`. If its interval spans zero,
then pre-training on pseudo-labels does no better than pre-training on the same images with
those labels shuffled — and whatever the first version measured was budget and exposure, not
the information the clustering had found.
"""),
        code("""
from mri_semisupervised.viz.plots import plot_roc_compare

curves = {
    arm: (group["y_true"].to_numpy(), group["y_score"].to_numpy())
    for arm, group in predictions.groupby("arm")
}
plot_roc_compare(curves, title="Pooled out-of-fold ROC, five repeats")
"""),
        md("""
## 4. Which clustering method each fold picked

Choosing the method inside the fold means the choice can differ from one fold to the next.
That is not instability to hide — it is a measurement of how much the choice depended on
seeing every label.
"""),
        code("""
print(folds["pseudo_method"].value_counts().to_string())
print()
print(f"ARI on the training labels: {folds['pseudo_ari_on_train'].mean():.3f} "
      f"+/- {folds['pseudo_ari_on_train'].std():.3f}")
print(f"pseudo-labels per fold    : {folds['n_pseudo'].mean():.0f}")
"""),
        md("""
## 5. What the leaks were worth

The `legacy` run reproduces the original protocol faithfully — duplicates left in place,
method and alignment decided once over every label, checkpoint taken on training accuracy.
It exists so the difference is **measured** rather than quoted from an old file.
"""),
        code("""
legacy_runs = [p for p in runs if p.name.startswith("legacy")]
if legacy_runs:
    legacy = pd.read_parquet(legacy_runs[-1] / "per_fold.parquet")
    comparison = pd.concat(
        [
            per_fold.groupby("arm")[headline].mean().add_suffix("_corrected"),
            legacy.groupby("arm")[headline].mean().add_suffix("_legacy"),
        ],
        axis=1,
    )
    display(comparison.round(3))
else:
    print("no legacy run found: uv run python scripts/run_experiment.py --mode legacy")
"""),
        md("""
## 6. What to take away

Whatever the numbers above say, the method is the point:

* an identity that comes from the data, so a duplicate cannot hide behind a folder;
* every decision that reads a label made inside the training fold;
* a control arm that makes "it helped" a falsifiable claim rather than a hopeful one;
* intervals, because with twenty images per fold a difference of one image moves recall by
  0.05.

A protocol that can only confirm what you hoped is not a protocol.
"""),
    ]


def main() -> None:
    write_notebook(NOTEBOOKS_DIR / "01_dataset_and_clustering.ipynb", build_dataset_notebook())
    write_notebook(NOTEBOOKS_DIR / "02_protocol_and_results.ipynb", build_protocol_notebook())


if __name__ == "__main__":
    main()
