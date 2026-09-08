# corrected-20260903-235109

Protocol: **corrected**

## Per-arm means across folds

| arm | roc_auc | pr_auc | recall_positive | f1_macro | accuracy |
|---|---|---|---|---|---|
| permuted_control | 0.921 +/- 0.081 | 0.931 +/- 0.070 | 0.816 +/- 0.191 | 0.820 +/- 0.108 | 0.824 +/- 0.105 |
| semi_supervised | 0.926 +/- 0.075 | 0.943 +/- 0.053 | 0.876 +/- 0.133 | 0.866 +/- 0.077 | 0.869 +/- 0.075 |
| supervised | 0.952 +/- 0.051 | 0.950 +/- 0.057 | 0.896 +/- 0.121 | 0.862 +/- 0.058 | 0.865 +/- 0.055 |

## Pooled out-of-fold, with bootstrap intervals

| arm | ROC AUC | PR-AUC |
|---|---|---|
| permuted_control | 0.911 [0.884, 0.935] | 0.907 [0.871, 0.940] |
| semi_supervised | 0.923 [0.897, 0.946] | 0.932 [0.904, 0.956] |
| supervised | 0.908 [0.881, 0.932] | 0.905 [0.874, 0.932] |

## Paired differences across the shared folds

| comparison | metric | difference | 95% CI | p |
|---|---|---|---|---|
| supervised vs semi_supervised | roc_auc | +0.026 | [-0.002, +0.055] | 0.084 |
| supervised vs permuted_control | roc_auc | +0.031 | [+0.006, +0.061] | 0.003 |
| semi_supervised vs permuted_control | roc_auc | +0.005 | [-0.021, +0.034] | 0.707 |
| supervised vs semi_supervised | recall_positive | +0.020 | [-0.032, +0.072] | 0.423 |
| supervised vs permuted_control | recall_positive | +0.080 | [+0.020, +0.140] | 0.012 |
| semi_supervised vs permuted_control | recall_positive | +0.060 | [-0.012, +0.132] | 0.103 |

## Clustering method chosen, fold by fold

- `Agglomerative(ward)`: 24 fold(s)
- `KMeans`: 1 fold(s)

ARI on the training labels: 0.468 ± 0.145
