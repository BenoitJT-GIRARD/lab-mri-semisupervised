# Semi-supervised MRI — a protocol that can say no

100 labelled brain MRIs, 1 406 unlabelled ones, and a question worth asking: when labels
are the scarce resource, can clustering the unlabelled pool into pseudo-labels buy
anything?

The first version of this project answered yes. This one answers **no** — and then spends
most of its effort establishing that the answer is about the method and not about a broken
experiment. Six variants of semi-supervision were tried, at four labelling budgets, each
against a control that isolates information from exposure.

![Label efficiency: the task, and then the question](reports/figures/label_efficiency.png)

## Project status

**Deliberately finished.** Built, audited, rebuilt around the audit, then audited again by
two outside readers and rebuilt around what survived arbitration. Not maintained beyond
that, and its CI does not run on a schedule.

Concretely:

- every number below comes from a run whose manifest is in `reports/experiments/`, naming
  its dataset fingerprint, seeds, package versions, GPU and git revision;
- the conclusion is **negative**, and it is published as it came out;
- what remains unproven is at the end, not buried.

## The result

Twenty-five folds — five stratified folds, five repeats — on 99 distinct evaluation
images, 59 training labels per fold. Eight arms, all sharing folds, architecture and
starting weights.

| arm | what it adds | ROC AUC |
|---|---|---|
| `supervised` | nothing — the reference | **0.966** ± 0.035 |
| `self_training` | pseudo-labels from its own first pass | 0.963 |
| `semi_supervised` | pseudo-labels from the fold's clustering | 0.958 ± 0.041 |
| `semi_supervised_joint` | the same, present at every step instead of as a phase | 0.955 |
| `permuted_control` | the same images, **labels shuffled** | 0.952 ± 0.063 |
| `self_training_control` | self-training's images, labels shuffled | 0.948 |
| `semi_supervised_confident` | only the pseudo-labels above a confidence cut | 0.945 ± 0.058 |
| `joint_permuted_control` | joint training on shuffled labels | 0.922 |

**Every arm that touches the unlabelled pool sits below the baseline that ignores it.**
Not significantly — the intervals overlap — but consistently, at every budget, under every
variant.

### The pseudo-labels do carry information

Against its own control, joint training wins clearly: **+0.033 [+0.009, +0.057], p = 0.008**
at 59 labels, and **+0.113 [+0.050, +0.180], p = 0.002** at 10. The clustering is finding
something real, and something a shuffle destroys.

Against the baseline it still loses: −0.010 and −0.016, neither significant. So the finding
is sharper than "semi-supervision does not work here":

> The pseudo-labels carry real information. Injecting it costs more than it is worth.

### One number that means the opposite of what it looks like

That `p = 0.002` is the trap this repository exists to avoid, and it is worth walking
through. The joint control sits at **0.811** — training on shuffled labels at every step is
catastrophic, far worse than not pre-training at all. So `+0.113` measures how much damage
the shuffle does, not how much the real labels add. Read against the control it looks like
a win; read against the baseline the same arm is behind.

A control tells you what a treatment is worth only if you also ask what doing nothing is
worth.

### The curve, and what it rules out

If pseudo-labels help anywhere, they help when labels are scarce. So the arms were rerun at
10, 20 and 40 training labels per fold, with the pool left whole — it is the resource under
test — and the clustering capped to the same labels as the fine-tuning, so the
semi-supervised arm never carries more label information than its own baseline.

| training labels | `supervised` | `semi_supervised` | `permuted_control` | semi − control |
|---|---|---|---|---|
| 10 | 0.941 | 0.941 | 0.920 | +0.022 [−0.005, +0.051] · p = 0.12 |
| 20 | 0.952 | 0.926 | 0.921 | +0.005 [−0.021, +0.034] · p = 0.71 |
| 40 | 0.957 | 0.947 | 0.947 | +0.000 [−0.022, +0.025] · p = 0.97 |
| 59 | 0.966 | 0.958 | 0.952 | +0.006 [−0.012, +0.029] · p = 0.57 |

Nothing significant at any budget, before or after Holm over the family. The largest effect
sits at the smallest budget, which is the direction the literature predicts, and its
interval still spans zero.

## Why it comes out that way

Two measurements explain the result rather than repeating it.

**The clustering was separating images by contrast.** Running the whole protocol again with
histogram equalisation in front of it improves the supervised baseline by
**+0.013 [+0.003, +0.024], p = 0.006** — the only comparison in this project that clears
conventional significance — and collapses the clustering's agreement with the labels from
an ARI of **0.457 ± 0.096** to **0.205 ± 0.179**. Remove the global intensity differences
and the structure the clustering was finding goes with them. On this dataset contrast
happens to correlate with the class; it is not what a radiologist would be looking at.

Equalisation stays **off by default**. Turning it on would buy 1.3 points of AUC on a
classifier this repository does not ship, while degrading the object it studies. Both runs
are published, so the trade is visible rather than decided for the reader.

**And the task is nearly saturated.** The supervised baseline reaches 0.966 with **8 of 25
folds already at exactly 1.000**. That leaves 0.034 of headroom, against a fold-to-fold
standard deviation of 0.035 and paired intervals about 0.020 wide. A gain would have to
close more than half of what remains to be visible at all.

The curve says the same thing from the other side: the marginal return is **0.11 AUC points
per hundred labels** between 10 and 20, and **0.03** between 20 and 40. Still positive at
59 — labelling more would still pay — but flattening.

An ImageNet-pretrained ResNet18, fine-tuned on 59 images, has very little left to be helped
with. That is a property of this dataset and this backbone, and it is the most transportable
thing here: it tells a reader how many images they would need to annotate before the
semi-supervised question even arises.

## What the first version got wrong

Three leaks, all pointing the same way, plus a confound.

**A third of the evaluation set was in the training pool.** `image_id` hashed the *path*, so
two copies of one scan in two folders got two identifiers, and the guard meant to keep
labelled images out of the pool never fired. 31 of the 99 distinct evaluation images were
in the pool the semi-supervised arm pre-trained on.

**The clustering method was chosen using the test labels**, and so was the cluster-to-class
alignment: both were decided once, over all 100 labels, outside the folds. Every test fold
helped pick the method that would later be tested on it.

**The comparison confounded method with budget** — the semi-supervised arm simply took more
gradient steps than its baseline.

**And the reported gain was a threshold move.** Recall on the positive class rose from 0.900
to 0.960 *while ROC AUC fell* from 0.982 to 0.954. A model that ranks better cannot do that;
what had moved was the implicit 0.5 cut.

`reports/experiments/legacy/` reproduces those three leaks and nothing else, on the same
machinery, so the difference prices them. One detail is worth the trouble: the pseudo-label
ARI is **0.481 ± 0.000** under the leak and **0.457 ± 0.096** once computed per fold. The
leak did not only inflate a number — it made it look far more stable than it is.

![What the leaks were worth](reports/figures/leak_price.png)

## What replaced it

- **Identity from content.** SHA-256 of the decoded pixels. The duplicate rules are written
  down because they are decisions: a leaked copy leaves the *pool* and never the evaluation
  set, a repeated evaluation image is reduced to one, and the same content under two labels
  stops the build rather than being quietly resolved.
- **The fold is a boundary.** Clustering, method selection and alignment all happen inside
  the training fold. The test fold takes part in no decision — not the checkpoint, not the
  threshold.
- **A control for every arm.** Each pre-training arm has a twin that sees the same images
  with the labels shuffled. It is what separates information from exposure, and it is the
  reason a null result here means something.
- **Honest model selection.** An inner validation split carved out of the training fold
  decides the checkpoint, the stopping point and the threshold. It stays at 20 images at
  every budget, so model selection is comparable across the curve.
- **Uncertainty first.** Paired comparisons across shared folds, bootstrap intervals over
  images, and Holm correction over the families of comparisons this project ended up
  running.

## What else was measured

**The probability scale is off, and by a lot.** Pooled out of fold, the supervised arm has a
Brier score of 0.104 and an expected calibration error of 0.101, with a mean score of 0.407
against a base rate of 0.505. The reliability curve says where: at a predicted 0.18 the
observed cancer rate is 0.46, at 0.44 it is 0.67, at 0.63 it is 0.86. **Mid-range scores
understate risk by twenty points or more.** Nothing is recalibrated — a network fine-tuned
on twenty images per fold has little chance of being calibrated, and measuring it and saying
so is the result.

![Calibration](reports/figures/calibration.png)

**The errors sit on the duplicates.** The 31 evaluation images that also live in the pool
are misclassified 0.155 of the time, against 0.068 for the other 68. The gap holds for the
*supervised* arm, which never sees the pool — so this is not contamination: those images are
simply harder. Being filed twice in the source folders correlates with being difficult,
which makes the original leak worse than it first looked. What leaked was not a random
sample; it was precisely the cases the model gets wrong.

**A sensitivity-first operating point does not transport.** Fixing a 90% recall floor on ten
validation positives yields **82.0%** realised recall on the test fold, and misses the floor
on 51 folds out of 100. It also sits above the F1 threshold on 95 folds out of 100 — trading
7 points of recall for 2 of precision. On this dataset the F1 threshold is the more sensitive
of the two, which is the reverse of what a screening frame wants. Both are published side by
side.

**The pseudo-labels over-call cancer.** 59.2% positive against a truth of 50.5%,
consistently (sd 0.063). The original protocol showed 66%; per-fold clustering removes two
thirds of the bias and not the rest.

## Running it

```powershell
uv sync --extra dev
$env:MRI_DATA_DIR = "C:\path\to\data"     # optional; defaults to ./data

uv run python scripts/build_manifest.py    # content identity, duplicates, fingerprint
uv run python scripts/build_features.py    # ResNet50 embeddings, cached
uv run python scripts/run_experiment.py    # the corrected protocol
uv run python scripts/run_experiment.py --mode legacy     # price the leaks
uv run python scripts/run_experiment.py --mode equalized  # the preprocessing variant
uv run python scripts/build_figures.py     # every figure above
```

One point of the curve, and the mechanism arms:

```powershell
uv run python scripts/run_experiment.py --label-budget 10 `
  --arms supervised,semi_supervised,permuted_control
uv run python scripts/run_experiment.py --arms semi_supervised_joint,joint_permuted_control `
  --out-name corrected-stageb
```

`MRI_DATA_DIR` exists because pre-training reads about 1 300 images per epoch: on a
synchronised drive that starves the GPU.

## Structure

```
├── src/mri_semisupervised/
│   ├── data/manifest.py     # identity from content, duplicate rules, fingerprint
│   ├── data/loader.py       # inventory and integrity
│   ├── features/            # frozen ResNet50 embeddings, cached and guarded
│   ├── models/              # clustering helpers, the shared classifier
│   └── protocol/            # everything that touches a label
│       ├── splits.py        # repeated stratified folds, inner split, label budgets
│       ├── pseudo_labels.py # clustering, selection, alignment and confidence, in the fold
│       ├── arms.py          # the eight arms
│       ├── evaluate.py      # threshold-free metrics and two operating points
│       ├── calibration.py   # reliability curves, Brier, ECE
│       ├── errors.py        # which images are missed, and whether they have anything in common
│       ├── uncertainty.py   # bootstrap intervals, paired comparisons, Holm
│       └── experiment.py    # orchestration and the run manifest
├── notebooks/               # generated by scripts/build_notebooks.py — see below
├── reports/experiments/     # per-fold metrics, raw predictions, run manifests
└── tests/                   # 148 tests, of which 9 assert the protocol cannot leak
```

The split between `models/` and `protocol/` is the point. What is *modelled* and what is
*decided* used to live in one module, and that is how a leak came to look like an ordinary
line.

## The tests that assert the absence of leakage

```powershell
uv run python -m pytest
```

Eight watch the orchestration: no test image in any pre-training set, no labelled image
either, the copies of evaluation images never reaching it, the inner validation never drawn
from the test fold, every arm of a fold sharing one split, each control pre-training on
exactly the same images as the arm it controls.

A ninth asserts the opposite — the `legacy` mode **must** pre-train on the copies of its own
test images. Without it the other eight could pass on a protocol that never had a defect,
and would prove nothing.

The same pairing guards the alignment: one test flips the labels the mask hides and checks
that nothing moves; another shows those same labels *would* have changed the answer if they
had been read.

Two more are worth naming because they caught real defects. One asserts that no implemented
arm can be written and then never run — a fourth arm was once added to the registry and not
to the configuration, and the experiment ran three arms while every test passed. Another
records that the joint arm carries a second treatment nobody intended: its pseudo batches
pass through the network in train mode, so BatchNorm absorbs the pool's statistics even at
weight zero. That is why the joint arm is only ever read against its own control.

## A note on the notebooks

The two notebooks are **generated**, by `uv run python scripts/build_notebooks.py`. Editing
a `.ipynb` by hand works until the next generation overwrites it; the source of every cell
is in that script.

It is built that way for three reasons: the cell text is versioned and diffable instead of
being buried in JSON, the imports cannot drift away from the `mri_semisupervised` package
because they are written against it, and `nbconvert --execute` replays them with no manual
step — so a stored output is always a capture of a real run.

## What this still does not prove

**That semi-supervised learning cannot help here.** Six variants were tried and none beat
the baseline, which is a much stronger statement than one variant would support — but it is
still about clustering-derived and self-derived pseudo-labels on ImageNet embeddings.
Self-supervised pre-training on the pool itself was not tried, and it is the one approach
with a real claim on 1 400 unlabelled images.

**Anything at n = 99.** Twenty images per fold; one image moves recall by 0.05. Every
interval above is wide because the data is small, and no protocol fixes that. The headroom
measurement makes the limit explicit rather than leaving it implied.

**Generalisation beyond this dataset.** One source, one modality, 512×512 slices, no patient
identifiers — so grouping by patient, the first thing a medical dataset needs, cannot even
be checked here. That is a limitation of the data, and it would be the first question to ask
of any real deployment.

## Licence and data

MIT, for the code.

**The MRI images are not redistributed here, and this repository does not hold the right
to redistribute them.** They are third-party medical images obtained as a fixed archive;
their terms are not public, so anyone reproducing this supplies their own copy and points
`MRI_DATA_DIR` at it.

What *is* versioned is derived and carries no image data: fold indices, per-fold metrics,
out-of-fold predictions as scores, and the manifests under `reports/experiments/` that
record the dataset fingerprint, the seeds, the package versions and the git revision behind
each run. That is enough to check every number in this README and not enough to reconstruct
a single scan.

The dataset fingerprint in each manifest is what lets someone with the same archive confirm
they are looking at the same 1 506 images.
