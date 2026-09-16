# Engineering decisions, and what was deliberately left out

Only the choices that could have gone the other way are here. The protocol itself is in
[`protocol.md`](protocol.md).

## What is modelled, and what is decided

```
src/mri_semisupervised/
├── data/         identity from content, inventory, preprocessing
├── features/     frozen ResNet50 embeddings, cached and guarded
├── models/       clustering helpers, the shared classifier
└── protocol/     everything that touches a label
```

The split between `models/` and `protocol/` is the one that matters. A clustering call and a
decision about which clustering to trust used to sit in the same module, which is how a leak
came to look like an ordinary line of code. Everything that reads a label now lives under
`protocol/`, where a reviewer can find it: the splits, the pseudo-labels, the arms, the
metrics, the intervals, the orchestration.

## The embeddings are frozen, and cached

The backbone is a ResNet50 with ImageNet weights, used as a feature extractor and never
trained. Each image is embedded once and the result is cached as a Parquet file, because the
same embeddings feed every clustering candidate of every fold; recomputing them per fold
would multiply a twenty-minute pass by twenty-five.

The cache is guarded. It records the backbone it was built with and whether histogram
equalisation was applied, and it refuses to serve a run that asks for anything else: an
equalised cache handed to a plain run would silently change the pixels behind every number
downstream. Equalisation gets its own cache file for the same reason.

## The classifier is small on purpose

A ResNet18, fine-tuned from ImageNet weights, with a shared starting point across every arm
and an identical number of gradient steps. It is the part of the system that matters least:
what is under test is the pre-training, so the classifier is held constant everywhere and
chosen small enough that 25 folds × 8 arms fit in an evening on one consumer GPU.

## Two PyTorch indexes, and why

The experiments train on an NVIDIA card under Windows, so `torch` and `torchvision` resolve
against the CUDA 12.6 index there; continuous integration runs on Linux with no GPU, where
the same pin would pull about three gigabytes of NVIDIA runtime to execute a CPU test suite.
The marker is the platform and not the presence of a card, because `uv` resolves the lock
file ahead of time and cannot probe the hardware.

## The notebooks are generated

`scripts/build_notebooks.py` writes both `.ipynb` files. The cell text is versioned and
diffable as Python instead of being buried in JSON, the imports cannot drift away from the
package because they are written against it, and `scripts/run_notebooks.py` replays them
from a clean kernel so the stored outputs are the capture of one execution.

## What was deliberately not done

**No self-supervised pre-training on the pool.** SimCLR or a masked autoencoder over 1 300
unlabelled slices is the approach with a real claim on this data, and it is out of scope
here: the question asked is whether *clustering-derived* pseudo-labels pay, and answering a
different question would not answer that one.

**No recalibration.** The probability scale is measurably off, and it stays off: a network
fine-tuned on twenty images per fold has little chance of being calibrated, and measuring the
gap is the result. Fitting a calibrator on this much data would mostly fit its own noise.

**No histogram equalisation by default.** It buys 1.3 points of supervised AUC and destroys
the structure the clustering was finding, which is the object under study. Both runs are
published; the reader decides.

**No patient-level grouping.** The archive carries no identifier of any kind, so the folds
group by image and nothing can be done about it. [`data-source.md`](data-source.md) says what
that rules out.

**No service, no container, no interface.** The deliverable is a measurement and the evidence
behind it. A dashboard over 99 evaluation images would be decoration, and a container around
a model this repository does not recommend deploying would be worse.
