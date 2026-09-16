# corrected-20260904-001811

Protocol: **corrected**

- dataset fingerprint: `8f22a69ffa46ca1f`
- evaluation images: 99
- folds: 25 (5 splits x 5 repeats)
- unlabelled pool: 1311

## Per-arm means across folds

| arm | roc_auc | pr_auc | recall_positive | f1_macro | accuracy |
|---|---|---|---|---|---|
| permuted_control | 0.947 +/- 0.044 | 0.954 +/- 0.036 | 0.860 +/- 0.155 | 0.853 +/- 0.083 | 0.856 +/- 0.080 |
| semi_supervised | 0.947 +/- 0.048 | 0.956 +/- 0.040 | 0.880 +/- 0.150 | 0.866 +/- 0.091 | 0.868 +/- 0.088 |
| supervised | 0.957 +/- 0.040 | 0.959 +/- 0.039 | 0.908 +/- 0.175 | 0.865 +/- 0.109 | 0.871 +/- 0.099 |

## Pooled out-of-fold, with bootstrap intervals

| arm | ROC AUC | PR-AUC |
|---|---|---|
| permuted_control | 0.933 [0.910, 0.955] | 0.933 [0.903, 0.958] |
| semi_supervised | 0.939 [0.916, 0.960] | 0.947 [0.923, 0.966] |
| supervised | 0.934 [0.912, 0.954] | 0.928 [0.894, 0.955] |

## Paired differences across the shared folds

| comparison | metric | difference | 95% CI | p |
|---|---|---|---|---|
| supervised vs semi_supervised | roc_auc | +0.010 | [-0.011, +0.029] | 0.379 |
| supervised vs permuted_control | roc_auc | +0.010 | [-0.009, +0.029] | 0.315 |
| semi_supervised vs permuted_control | roc_auc | +0.000 | [-0.022, +0.025] | 0.970 |
| supervised vs semi_supervised | recall_positive | +0.028 | [-0.060, +0.108] | 0.468 |
| supervised vs permuted_control | recall_positive | +0.048 | [-0.016, +0.112] | 0.135 |
| semi_supervised vs permuted_control | recall_positive | +0.020 | [-0.048, +0.084] | 0.615 |

## Clustering method chosen, fold by fold

- `Agglomerative(ward)`: 25 fold(s)

ARI on the training labels: 0.436 ± 0.113
