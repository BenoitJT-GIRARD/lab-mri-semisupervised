"""Semi-supervised learning on brain MRI, under a protocol that cannot leak.

The package is in two halves, and the split is the point:

* :mod:`mri_semisupervised.data`, :mod:`~mri_semisupervised.features` and
  :mod:`~mri_semisupervised.models` hold what is *modelled* — identity, embeddings,
  clustering, the classifier;
* :mod:`mri_semisupervised.protocol` holds what is *decided* — which fold sees which
  image, which labels may be read, where the checkpoint and the threshold come from.

They used to be mixed, and that is how a leak came to look like an ordinary line.
"""

from __future__ import annotations

__version__ = "0.1.0"
