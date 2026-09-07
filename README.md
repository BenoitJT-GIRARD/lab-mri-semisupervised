# Semi-supervised MRI — a protocol that can say no

100 labelled brain MRIs, 1 406 unlabelled ones, and a question worth asking: can clustering
the unlabelled pool into pseudo-labels buy anything, when labels are the scarce resource?

The first version of this project answered yes. This one answers **no**, and the interesting
part is why the first answer did not survive.

![Paired differences between the three arms](reports/figures/paired_differences.png)

## Project status

**Deliberately finished.** Built, then audited, then rebuilt around the audit's findings.
It is not maintained beyond that, and its CI does not run on a schedule.

What that means concretely:

- every figure and every number below comes from one of two runs whose manifests are in
  `reports/experiments/`, each naming its dataset fingerprint, its seeds, its package
  versions and its git revision;
- the conclusion is **negative**, and it is published as it came out;
- what remains unproven is listed at the end, not buried.

## What the audit found

### 1. A third of the evaluation set was in the training pool

The dataset ships 1 506 files. Hashing their **contents** shows 1 410 distinct images:

| | |
|---|---|
| Labelled files | 100 |
| **Distinct evaluation images** | **99** — one image is filed twice |
| Real class balance | **50 cancer / 49 normal**, not the announced 50/50 |
| Evaluation images also present in the unlabelled pool | **31 of 99 — 31%** |
| of which `normal` / `cancer` | 23 / 8 |
| Redundant copies inside the unlabelled pool | 63 |

The semi-supervised arm pre-trains on that pool. A third of every test fold was therefore
already seen, whatever the cross-validation did — and the leak is **asymmetric**, so it does
not add noise, it leans.

**The code had tried to prevent exactly this.** It excluded the labelled images from the
pseudo-labels, with a comment saying so. But the exclusion tested the *folder*:

```python
weak = df[df["split"] == "unlabeled"][...]
```

while an image's identity was:

```python
def _safe_hash(path: Path) -> str:
    return hashlib.md5(str(path).encode("utf-8"), ...).hexdigest()[:12]
```

`image_id` hashed the **path**. Two copies of one scan in two folders got two identifiers,
and the guard had nothing to hold on to.

*Transferable lesson: a guard that rests on where a file sits protects nothing. Identity has
to come from the data.*

### 2. The clustering method was chosen using the test labels

```python
best = max((r for r in results if r.ari_vs_truth is not None), key=lambda r: r.ari_vs_truth)
```

Five algorithms, and the one producing the pseudo-labels was picked by its ARI against all
100 labels — every test fold included. The spread between candidates was not cosmetic:
0.60 for `Agglomerative(ward)` against 0.07 for K-Means.

### 3. So was the cluster-to-class alignment

Each cluster took the majority true label of the labelled images inside it, over all 100,
once, outside the folds. Every test fold helped decide the polarity of the pseudo-labels
used to pre-train the model then evaluated on it.

### 4. And the comparison confounded method with budget

The semi-supervised arm ran 3 epochs on ~1 400 images **plus** 6 on the 80 training ones;
its baseline ran 6. More steps, more images, and no control.

### 5. The reported gain was a threshold move, not better ranking

The published numbers had recall rising from 0.900 to 0.960 while **ROC AUC fell** from
0.982 to 0.954. A model that ranks better cannot do that. At 10 cancer images per fold, the
whole gain was one image.

## What replaced it

* **Identity from content.** SHA-256 of the decoded pixels. The duplicate rules are written
  down because they are decisions: a leaked copy leaves the *training pool* and never the
  evaluation set, a repeated evaluation image is reduced to one, and the same content under
  two labels stops the build rather than being quietly resolved.
* **The fold is a boundary.** Clustering, method selection and alignment all happen inside
  the training fold. The test fold takes part in no decision — not the checkpoint, not the
  threshold.
* **Three arms**, sharing folds, architecture and starting weights:

| arm | pre-training | what it isolates |
|---|---|---|
| `supervised` | none | the reference |
| `semi_supervised` | the fold's pseudo-labels | the supposed contribution |
| `permuted_control` | **the same images, labels shuffled** | budget and exposure |

* **Honest model selection.** An inner validation split carved out of the training fold
  decides the checkpoint, the stopping point and the decision threshold.
* **Uncertainty first.** 5 folds × 5 repeats, paired comparisons across shared folds,
  bootstrap intervals over images.

The third arm is what makes the result mean something. It sees the same images and takes
the same gradient steps; only the pairing between an image and its pseudo-label is
destroyed.

## What the corrected protocol says

25 folds, three arms, dataset fingerprint `8f22a69ffa46ca1f`:

| arm | ROC AUC | PR-AUC | recall (cancer) | F1 macro |
|---|---|---|---|---|
| supervised | **0.966** ± 0.035 | 0.963 ± 0.043 | **0.920** ± 0.112 | **0.904** ± 0.066 |
| semi-supervised | 0.958 ± 0.041 | **0.964** ± 0.034 | 0.888 ± 0.120 | 0.881 ± 0.075 |
| permuted control | 0.952 ± 0.063 | 0.958 ± 0.051 | 0.884 ± 0.128 | 0.895 ± 0.067 |

Paired across the shared folds, with 95% bootstrap intervals:

| comparison | ROC AUC | recall (cancer) |
|---|---|---|
| semi-supervised − permuted control | +0.006 [−0.012, +0.029] · p = 0.57 | +0.004 [−0.048, +0.052] · p = 0.86 |
| semi-supervised − supervised | −0.007 [−0.022, +0.007] · p = 0.35 | −0.032 [−0.084, +0.024] · p = 0.24 |

**The semi-supervised arm is indistinguishable from a control pre-trained on the same
images with the labels shuffled.** Every interval spans zero. Whatever pre-training bought,
it was exposure to the images, not the information the clustering found.

The plain supervised baseline is nominally ahead of both — also within the noise. On 99
images, that is the honest resolution of this experiment.

![The three arms](reports/figures/arms_comparison.png)

## What the leaks were worth

`reports/experiments/legacy/` reproduces **the three leaks and nothing else** — duplicates
left in place, method and alignment decided once over every label — on the same training
machinery, so the difference prices the leaks rather than confounding them with the other
repairs.

The result is not the one that was expected. Even with the leaks in place, the
semi-supervised arm does not reproduce the published advantage: it comes out **worse** than
its baseline on recall, by 0.048 [+0.004, +0.096], p = 0.034.

So the original conclusion does not survive its own protocol either. What produced it was
most likely the combination of a checkpoint chosen on training accuracy and a fixed 0.5
threshold, on folds of twenty images.

One detail worth the trouble: the ARI of the pseudo-labels is **0.481 ± 0.000** under the
leak and **0.457 ± 0.096** once it is computed per fold. The leak did not only inflate a
number, it made it look far more stable than it is.

![What the leaks were worth](reports/figures/leak_price.png)

## Running it

```powershell
uv sync --extra dev
$env:MRI_DATA_DIR = "C:\path\to\data"     # optional; defaults to ./data

uv run python scripts/build_manifest.py    # content identity, duplicates, fingerprint
uv run python scripts/build_features.py    # ResNet50 embeddings, cached
uv run python scripts/run_experiment.py    # the corrected protocol
uv run python scripts/run_experiment.py --mode legacy   # price the leaks
uv run python scripts/build_figures.py     # the figures above
```

`MRI_DATA_DIR` exists because the pre-training reads about 1 300 images per epoch: on a
synchronised drive that starves the GPU.

## Structure

```
├── src/mri_semisupervised/
│   ├── data/manifest.py     # identity from content, duplicate rules, fingerprint
│   ├── data/loader.py       # inventory and integrity
│   ├── features/            # frozen ResNet50 embeddings, cached
│   ├── models/              # clustering helpers, the shared classifier
│   └── protocol/            # everything that touches a label
│       ├── splits.py        # repeated stratified folds, inner split, derived seeds
│       ├── pseudo_labels.py # clustering, selection and alignment, inside the fold
│       ├── arms.py          # the three arms
│       ├── evaluate.py      # threshold-free and at-threshold metrics
│       ├── uncertainty.py   # bootstrap intervals, paired comparisons
│       └── experiment.py    # orchestration and the run manifest
├── notebooks/               # the dataset, then the protocol and its verdict
├── reports/experiments/     # per-fold metrics, raw predictions, run manifests
└── tests/                   # 71 tests, of which 8 assert the protocol cannot leak
```

The split between `models/` and `protocol/` is the point. What is *modelled* and what is
*decided* used to live in one module, and that is how a leak came to look like an ordinary
line.

## Tests that assert the absence of leakage

```powershell
uv run python -m pytest
```

Eight of them watch the orchestration: no test image in any pre-training set, no labelled
image either, the copies of evaluation images never reaching it, the inner validation never
drawn from the test fold, the three arms of a fold sharing one split, the control
pre-training on exactly the same images as the semi-supervised arm.

A ninth asserts the opposite — the `legacy` mode **must** pre-train on the copies of its own
test images. Without it the other eight could pass on a protocol that never had a defect,
and would prove nothing.

The same pairing guards the alignment: one test flips the labels the mask hides and checks
nothing moves; another shows those same labels would have changed the answer if read.

## What this still does not prove

**That semi-supervised learning does not help here.** It shows this clustering, on these
embeddings, at ARI ≈ 0.46, does not beat shuffled labels. A better representation might. The
experiment answers the question it was given, not the general one.

**Anything at n = 99.** Twenty images per fold; one image moves recall by 0.05. Every
interval published above is wide because the data is small, and no amount of protocol fixes
that.

**Generalisation beyond this dataset.** One source, one modality, 512×512 slices, no
patient identifiers — so grouping by patient, which is the first thing a medical dataset
needs, cannot even be checked here. That is a limitation of the data, and it would be the
first question to ask of any real deployment.

## Licence

MIT
