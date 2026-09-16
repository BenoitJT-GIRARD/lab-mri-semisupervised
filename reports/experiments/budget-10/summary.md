# corrected-20260903-232508

Written by `scripts/rebuild_summaries.py` from the artefacts of the run of 2026-09-03.

Protocol: **corrected**

- dataset fingerprint: `8f22a69ffa46ca1f`
- evaluation images: 99
- folds: 25 (5 splits x 5 repeats)
- unlabelled pool: 1311

## Per-arm means across folds

| arm | roc_auc | pr_auc | recall_positive | f1_macro | accuracy |
|---|---|---|---|---|---|
| permuted_control | 0.920 +/- 0.073 | 0.940 +/- 0.057 | 0.808 +/- 0.182 | 0.837 +/- 0.091 | 0.842 +/- 0.080 |
| semi_supervised | 0.941 +/- 0.047 | 0.951 +/- 0.044 | 0.852 +/- 0.142 | 0.873 +/- 0.084 | 0.874 +/- 0.082 |
| supervised | 0.941 +/- 0.039 | 0.949 +/- 0.035 | 0.896 +/- 0.084 | 0.877 +/- 0.081 | 0.879 +/- 0.077 |

## Pooled out-of-fold, with bootstrap intervals

| arm | ROC AUC | PR-AUC |
|---|---|---|
| permuted_control | 0.896 [0.867, 0.924] | 0.907 [0.877, 0.936] |
| semi_supervised | 0.934 [0.910, 0.955] | 0.927 [0.891, 0.958] |
| supervised | 0.870 [0.840, 0.900] | 0.875 [0.836, 0.909] |

## Paired differences across the shared folds

| comparison | metric | difference | 95% CI | p |
|---|---|---|---|---|
| supervised vs semi_supervised | roc_auc | -0.001 | [-0.022, +0.018] | 0.945 |
| supervised vs permuted_control | roc_auc | +0.021 | [-0.005, +0.047] | 0.119 |
| semi_supervised vs permuted_control | roc_auc | +0.022 | [-0.005, +0.051] | 0.118 |
| supervised vs semi_supervised | recall_positive | +0.044 | [-0.004, +0.092] | 0.081 |
| supervised vs permuted_control | recall_positive | +0.088 | [+0.024, +0.156] | 0.006 |
| semi_supervised vs permuted_control | recall_positive | +0.044 | [-0.028, +0.116] | 0.248 |

## Clustering method chosen, fold by fold

- `Agglomerative(ward)`: 24 fold(s)
- `KMeans`: 1 fold(s)

ARI on the training labels: 0.410 ± 0.177
