"""Generate the five notebooks programmatically, one per question.

Why the ``.ipynb`` files are built and not hand-written:

* the source of every cell lives in the repository, versioned, readable and diffable;
* the imports and the sections cannot drift away from the ``mri_semisupervised`` package,
  because they are written against it here;
* ``uv run python -m nbconvert --execute`` replays them with no manual step, so the stored
  outputs are always the capture of one real run.

Usage:
    uv run python scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from mri_semisupervised.config import PROJECT_ROOT as ROOT

NOTEBOOKS_DIR = ROOT / "notebooks"

# The notebooks take the root from the package too: `scripts/run_notebooks.py` runs them from
# the repository root, and a cell that recomputed it from the current directory would answer
# differently depending on where the kernel was started.
PREAMBLE = """
import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
"""

#: What a notebook that reads a finished run needs before its first table. The runs are read
#: from `reports/experiments/`, never recomputed: an evening of GPU does not belong in a cell.
RESULTS_PREAMBLE = """
import numpy as np
import pandas as pd

from mri_semisupervised.config import EXPERIMENTS_DIR
from mri_semisupervised.protocol.uncertainty import paired_difference

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

per_fold = pd.read_parquet(EXPERIMENTS_DIR / "corrected" / "per_fold.parquet")
predictions = pd.read_parquet(EXPERIMENTS_DIR / "corrected" / "predictions.parquet")
folds = pd.read_parquet(EXPERIMENTS_DIR / "corrected" / "folds.parquet")

#: The five metrics every table of these notebooks reports, in one order.
headline = ["roc_auc", "pr_auc", "recall_positive", "f1_macro", "accuracy"]
"""


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n"))


def cut(cells: list[nbf.NotebookNode], before: str) -> tuple[list, list]:
    """Split a list of cells at the section that opens with `before`.

    The five notebooks are five questions, and the prose that answers each of them was
    written as one run of sections. Cutting here keeps that prose where it is instead of
    copying it into five files that would drift apart.
    """
    index = next(i for i, cell in enumerate(cells) if cell.source.lstrip().startswith(before))
    return cells[:index], cells[index:]


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
from mri_semisupervised.config import LABELLED_DIR, UNLABELLED_DIR
from mri_semisupervised.data.manifest import apply_duplicate_rules, build_manifest, summarise

manifest = apply_duplicate_rules(build_manifest(LABELLED_DIR, UNLABELLED_DIR))
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
## 3. What the pixels look like, without showing them

Everything is 512x512 RGB. The two classes differ in their intensity distribution, which is
worth knowing before reading anything into a clustering that separates them.

**No scan appears in this notebook, and none can.** The archive allows academic use and not
redistribution, so what is plotted here is derived: the mean and the spread of each image,
never the image. `plot_image_grid` and `plot_equalization` exist in `viz/plots.py` for
looking at the data on a machine that holds it; their output is not committed anywhere.
"""),
        code("""
from mri_semisupervised.data.loader import compute_pixel_stats, discover_images
from mri_semisupervised.viz.plots import plot_pixel_stats

records, corrupted = discover_images()
print(f"{len(records)} readable files, {len(corrupted)} unreadable")

stats = compute_pixel_stats(records, sample_size=300, seed=0)
plot_pixel_stats(stats)
"""),
        code("""
by_class = stats.merge(
    manifest[["image_id", "label"]].drop_duplicates("image_id"), on="image_id", how="left"
)
by_class["label"] = by_class["label"].fillna("unlabelled")
by_class.groupby("label")[["mean", "std"]].describe().round(1).T
"""),
        md("""
The labelled classes sit a few grey levels apart on the mean, and the unlabelled pool covers
both. That gap is small, and notebook 04 shows what happens to the clustering when histogram
equalisation removes it.
"""),
        md("""
## 1. Clustering the embeddings

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

DBSCAN, at this eps, declares nearly everything noise. The number is reported as it came: a
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
## 2. Where this leaves the pseudo-labels

The best clustering reaches an ARI around 0.6 against the true labels. That is substantial
and far from decisive: the pseudo-labels it produces will be right often enough to be worth
trying, and wrong often enough that the trying has to be measured.

Notebook 03 measures it, under a protocol where the test fold takes part in no decision
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
        md("""
## 1. What the corrected protocol does differently

* Duplicates are removed **by content**, and the evaluation set never loses an image.
* The clustering, the choice of method and the alignment happen **inside the training
  fold**. The test fold takes part in no decision.
* Four arms share folds, architecture and starting weights:

| arm | pre-training | what it isolates |
|---|---|---|
| `supervised` | none | the reference |
| `semi_supervised` | the fold's pseudo-labels | the supposed contribution |
| `semi_supervised_confident` | only the pseudo-labels above a confidence cut | whether noise was the problem |
| `permuted_control` | the same images, labels shuffled | budget and exposure |

* The checkpoint and the decision threshold come from an inner validation split carved out
  of the training fold.
* Five repeats of a five-fold cross-validation, so the spread is measured and not
  guessed.

The control is the arm that can end the discussion. It sees the same images and takes the
same number of steps; only the pairing between an image and its pseudo-label is destroyed.
Every pre-training arm here has one.
"""),
        code("""
import json

corrected = EXPERIMENTS_DIR / "corrected"
print(f"reading {corrected.name}")

meta = json.loads((corrected / "manifest.json").read_text(encoding="utf-8"))
print(f"dataset fingerprint : {meta['dataset_fingerprint']}")
print(f"evaluation images   : {meta['evaluation_images']}")
print(f"unlabelled pool     : {meta['unlabelled_pool']}")
print(f"folds               : {meta['protocol']['n_splits']} x {meta['protocol']['n_repeats']} repeats")
print(f"gpu                 : {meta['versions']['gpu']}")

"""),
        md("""
## 2. The arms, side by side

Two families of number, and they answer different questions. **ROC AUC and PR-AUC** say how
well the scores rank, whatever threshold is applied. **Recall and F1** say what happens at
the threshold that was actually chosen — on the inner validation, never on the test fold.

The original comparison reported only the second kind, which is how a model whose ranking
had got *worse* came to look better.
"""),
        code("""
per_fold.groupby("arm")[headline].agg(["mean", "std"]).round(3)
"""),
        md("""
## 3. Paired comparisons

The arms share their folds, so the comparison is paired. Treating the two as independent
samples would throw the pairing away and widen every interval for nothing.
"""),
        code("""
pivot = per_fold.pivot(index="fold", columns="arm")
rows = []
for metric in ["roc_auc", "pr_auc", "recall_positive"]:
    for a, b in [
        ("semi_supervised", "supervised"),
        ("semi_supervised", "permuted_control"),
        ("semi_supervised_confident", "permuted_control"),
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
        md("""
## 1. Where the experiment could see anything at all

A null result is only worth reading if the experiment could have shown a gain. The
supervised baseline reaches 0.966 with eight folds out of twenty-five already at 1.000, so
what is left to win is about the size of the noise. The arms were therefore rerun at
smaller labelling budgets, where a semi-supervised method has room to help.
"""),
        code("""
budgets = {}
for name, size in [("budget-10", 10), ("budget-20", 20), ("budget-40", 40), ("corrected", 59)]:
    path = EXPERIMENTS_DIR / name / "per_fold.parquet"
    if path.exists():
        budgets[size] = pd.read_parquet(path).pivot(index="fold", columns="arm", values="roc_auc")

rows = []
for size, pivot in sorted(budgets.items()):
    row = {"labels": size}
    row.update({arm: round(pivot[arm].mean(), 3) for arm in pivot.columns})
    if {"semi_supervised", "permuted_control"} <= set(pivot.columns):
        out = paired_difference(
            pivot["semi_supervised"].to_numpy(), pivot["permuted_control"].to_numpy()
        )
        row["semi - control"] = f"{out['mean_difference']:+.3f} (p={out['p_value']:.2f})"
    rows.append(row)
pd.DataFrame(rows)
"""),
        md("""
Two things this table says that a single budget could not. The semi-supervised arm never
beats the plain baseline, at any budget. And the permuted control sits *below* that baseline
everywhere — so the pre-training phase costs something on its own, and real pseudo-labels
recover part of that cost without ever turning it into a gain.
"""),
        md("""
## 2. Two hypotheses about why, each with its own control

Two explanations for the null result are worth testing. Maybe the
pre-training is simply **forgotten** — 246 steps, then a fine-tuning that converges in two
to four epochs. Maybe the labels come from the **wrong source** — a k-means on ImageNet
embeddings, and not from the decision function being optimised.

`semi_supervised_joint` keeps the pseudo-label loss present at every step;
`self_training` builds its labels from its own first pass. Each is read against its own
control, never against the plain baseline.
"""),
        code("""
stage_b = EXPERIMENTS_DIR / "corrected-stageb" / "per_fold.parquet"
if stage_b.exists():
    mech = pd.read_parquet(stage_b).pivot(index="fold", columns="arm", values="roc_auc")
    joined = pd.concat([pivot_full := per_fold.pivot(index="fold", columns="arm",
                                                     values="roc_auc"), mech], axis=1)
    rows = []
    for arm, control in [("semi_supervised_joint", "joint_permuted_control"),
                         ("self_training", "self_training_control")]:
        against_control = paired_difference(joined[arm].to_numpy(), joined[control].to_numpy())
        against_baseline = paired_difference(joined[arm].to_numpy(),
                                             joined["supervised"].to_numpy())
        rows.append({
            "arm": arm,
            "vs its control": f"{against_control['mean_difference']:+.3f} "
                              f"(p={against_control['p_value']:.3f})",
            "vs supervised": f"{against_baseline['mean_difference']:+.3f} "
                             f"(p={against_baseline['p_value']:.3f})",
        })
    display(pd.DataFrame(rows))
"""),
        md("""
Joint training beats its control significantly and still loses to the baseline. Both facts
are needed: the control sits at 0.922 because training on shuffled labels at every step is
harmful, so the win against it measures that harm, and not a gain. A significant p-value
against the correct control can still mean the opposite of what it looks like.
"""),
        md("""
## 1. What a returned probability is worth

The protocol publishes scores. Whether they mean anything as probabilities is a separate
question, and ROC AUC cannot answer it: a model whose ranking is perfect and whose scale is
squashed scores 1.0 either way.
"""),
        code("""
from mri_semisupervised.protocol.calibration import summarise_calibration

summary, curves = summarise_calibration(predictions, n_bins=10)
display(summary.round(4))

sup = curves["supervised"]
display(sup[["mean_score", "observed", "gap"]].round(3))
"""),
        md("""
Read the `gap` column: where the model predicts 0.44 the observed cancer rate is 0.67. The
mid-range scores understate risk by twenty points and more, which is the direct reason an
operating point chosen on one split does not transport to another.

Nothing is recalibrated here. A network fine-tuned on twenty images per fold has little
chance of being calibrated, and measuring that and saying so is the result.
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
## 2. What the leaks were worth

The `legacy` run reproduces the original protocol faithfully — duplicates left in place,
method and alignment decided once over every label, checkpoint taken on training accuracy.
It exists so the difference is **measured** here, and not quoted from an old file.
"""),
        code("""
legacy_dir = EXPERIMENTS_DIR / "legacy"
if (legacy_dir / "per_fold.parquet").exists():
    legacy = pd.read_parquet(legacy_dir / "per_fold.parquet")
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
## 3. What to take away

Whatever the numbers above say, the method is the point:

* an identity that comes from the data, so a duplicate cannot hide behind a folder;
* every decision that reads a label made inside the training fold;
* a control arm that makes "it helped" a falsifiable claim;
* intervals, because with twenty images per fold a difference of one image moves recall by
  0.05.

A protocol that can only confirm what you hoped is not a protocol.
"""),
    ]


def main() -> None:
    dataset, clustering = cut(build_dataset_notebook(), "## 1. Clustering the embeddings")
    protocol, rest = cut(build_protocol_notebook(), "## 1. Where the experiment could see")
    efficiency, calibration = cut(rest, "## 1. What a returned probability is worth")

    write_notebook(NOTEBOOKS_DIR / "01_dataset.ipynb", dataset)
    write_notebook(
        NOTEBOOKS_DIR / "02_embeddings_and_clustering.ipynb",
        [
            md("""
# 02 — What the embeddings hold, and what a clustering can find in them

Notebook 01 established what the 1 506 files are. This one asks whether their ResNet50
embeddings carry the diagnosis at all: five clustering algorithms, the agreement of each one
with the labels, and the projection that shows what the space looks like.

Nothing here decides anything. Choosing a method on all the labels is exactly the mistake the
protocol was rebuilt to prevent, and notebook 03 shows how the choice is made inside a fold.
"""),
            code(PREAMBLE),
            *clustering,
        ],
    )
    write_notebook(
        NOTEBOOKS_DIR / "03_protocol_and_arms.ipynb",
        [
            md("""
# 03 — The protocol, and what the arms say under it

The comparison itself: what the corrected protocol changes, the eight arms side by side, the
paired differences across shared folds, and which clustering method each fold chose on its
own data.

Every table below reads a finished run from `reports/experiments/`. None of them trains
anything: a run is an evening of GPU, and a notebook that retrained would publish numbers
nobody could check against the manifests.
"""),
            code(RESULTS_PREAMBLE),
            *protocol,
        ],
    )
    write_notebook(
        NOTEBOOKS_DIR / "04_label_efficiency_and_mechanisms.ipynb",
        [
            md("""
# 04 — Where a gain could have shown, and why none did

Two questions that a single budget cannot answer. Could the experiment have seen a gain at
all, given how little headroom the task leaves? And if the pseudo-labels carry information,
what happens to it between the pre-training and the fine-tuning?

Each mechanism is read against its own control, never against the plain baseline.
"""),
            code(RESULTS_PREAMBLE),
            *efficiency,
        ],
    )
    write_notebook(
        NOTEBOOKS_DIR / "05_calibration_errors_and_leaks.ipynb",
        [
            md("""
# 05 — What a returned probability is worth, and what the leaks cost

Ranking is not everything. This notebook reads the probability scale the model returns, the
images it misses, and the price of the three leaks the first version of the protocol carried.
"""),
            code(RESULTS_PREAMBLE),
            *calibration,
        ],
    )


if __name__ == "__main__":
    main()
