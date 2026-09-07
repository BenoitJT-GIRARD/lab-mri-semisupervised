# corrected-20260903-123125

Protocol: **corrected**

## Per-arm means across folds

| arm | roc_auc | pr_auc | recall_positive | f1_macro | accuracy |
|---|---|---|---|---|---|
| permuted_control | 0.952 +/- 0.063 | 0.958 +/- 0.051 | 0.884 +/- 0.128 | 0.895 +/- 0.067 | 0.897 +/- 0.066 |
| semi_supervised | 0.958 +/- 0.041 | 0.964 +/- 0.034 | 0.888 +/- 0.120 | 0.881 +/- 0.075 | 0.883 +/- 0.073 |
| supervised | 0.966 +/- 0.035 | 0.963 +/- 0.043 | 0.920 +/- 0.112 | 0.904 +/- 0.066 | 0.905 +/- 0.063 |

## Pooled out-of-fold, with bootstrap intervals

| arm | ROC AUC | PR-AUC |
|---|---|---|
| permuted_control | 0.948 [0.929, 0.966] | 0.943 [0.913, 0.968] |
| semi_supervised | 0.937 [0.914, 0.958] | 0.943 [0.917, 0.963] |
| supervised | 0.948 [0.929, 0.966] | 0.947 [0.921, 0.969] |

## Paired differences across the shared folds

| comparison | metric | difference | 95% CI | p |
|---|---|---|---|---|
| supervised vs semi_supervised | roc_auc | +0.007 | [-0.007, +0.022] | 0.352 |
| supervised vs permuted_control | roc_auc | +0.014 | [-0.006, +0.037] | 0.220 |
| semi_supervised vs permuted_control | roc_auc | +0.006 | [-0.012, +0.029] | 0.571 |
| supervised vs semi_supervised | recall_positive | +0.032 | [-0.024, +0.084] | 0.243 |
| supervised vs permuted_control | recall_positive | +0.036 | [-0.020, +0.092] | 0.211 |
| semi_supervised vs permuted_control | recall_positive | +0.004 | [-0.048, +0.052] | 0.857 |

## Clustering method chosen, fold by fold

- `Agglomerative(ward)`: 24 fold(s)
- `GMM`: 1 fold(s)

ARI on the training labels: 0.457 ± 0.096
