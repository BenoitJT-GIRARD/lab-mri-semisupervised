"""Semi-supervised learning on brain MRI, under a protocol that cannot leak.

The package is in two halves, and the split is the point:

* :mod:`mri_semisupervised.data`, :mod:`~mri_semisupervised.features` and
  :mod:`~mri_semisupervised.models` hold what is *modelled* — identity, embeddings,
  clustering, the classifier;
* :mod:`mri_semisupervised.protocol` holds what is *decided* — the split, the labels a
  step is allowed to read, the epoch kept, the threshold applied.

They used to sit in the same modules, where a decision about labels read like any
other call.
"""

from __future__ import annotations

__version__ = "0.1.0"
