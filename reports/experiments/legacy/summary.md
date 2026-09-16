# legacy-20260903-120310

Written by `scripts/rebuild_summaries.py` from the artefacts of this run.

Protocol: **legacy**

- dataset fingerprint: `8f22a69ffa46ca1f`
- evaluation images: 100
- folds: 25 (5 splits x 5 repeats)
- unlabelled pool: 1342

## Per-arm means across folds

| arm | roc_auc | pr_auc | recall_positive | f1_macro | accuracy |
|---|---|---|---|---|---|
| permuted_control | 0.968 +/- 0.037 | 0.965 +/- 0.047 | 0.928 +/- 0.098 | 0.902 +/- 0.062 | 0.904 +/- 0.059 |
| semi_supervised | 0.959 +/- 0.045 | 0.953 +/- 0.061 | 0.880 +/- 0.135 | 0.898 +/- 0.066 | 0.900 +/- 0.063 |
| supervised | 0.964 +/- 0.038 | 0.963 +/- 0.045 | 0.928 +/- 0.084 | 0.911 +/- 0.060 | 0.912 +/- 0.058 |

## Pooled out-of-fold, with bootstrap intervals

| arm | ROC AUC | PR-AUC |
|---|---|---|
| permuted_control | 0.960 [0.941, 0.977] | 0.939 [0.900, 0.974] |
| semi_supervised | 0.954 [0.933, 0.973] | 0.938 [0.899, 0.971] |
| supervised | 0.947 [0.927, 0.965] | 0.947 [0.923, 0.968] |

## Paired differences across the shared folds

| comparison | metric | difference | 95% CI | p |
|---|---|---|---|---|
| supervised vs semi_supervised | roc_auc | +0.005 | [-0.008, +0.017] | 0.434 |
| supervised vs permuted_control | roc_auc | -0.004 | [-0.019, +0.010] | 0.607 |
| semi_supervised vs permuted_control | roc_auc | -0.009 | [-0.022, +0.002] | 0.155 |
| supervised vs semi_supervised | recall_positive | +0.048 | [+0.004, +0.096] | 0.034 |
| supervised vs permuted_control | recall_positive | +0.000 | [-0.028, +0.032] | 0.978 |
| semi_supervised vs permuted_control | recall_positive | -0.048 | [-0.104, +0.008] | 0.091 |

## Clustering method chosen, fold by fold

- `Agglomerative(ward)`: 25 fold(s)

ARI on the training labels: 0.481 ± 0.000
