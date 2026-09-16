# equalized-20260903-225735

Written by `scripts/rebuild_summaries.py` from the artefacts of the run of 2026-09-03.

Protocol: **equalized**

- dataset fingerprint: `8f22a69ffa46ca1f`
- evaluation images: 99
- folds: 25 (5 splits x 5 repeats)
- unlabelled pool: 1311

## Per-arm means across folds

| arm | roc_auc | pr_auc | recall_positive | f1_macro | accuracy |
|---|---|---|---|---|---|
| permuted_control | 0.968 +/- 0.031 | 0.968 +/- 0.033 | 0.916 +/- 0.172 | 0.891 +/- 0.096 | 0.895 +/- 0.089 |
| semi_supervised | 0.964 +/- 0.038 | 0.960 +/- 0.053 | 0.940 +/- 0.087 | 0.906 +/- 0.064 | 0.907 +/- 0.063 |
| semi_supervised_confident | 0.955 +/- 0.040 | 0.944 +/- 0.064 | 0.908 +/- 0.119 | 0.887 +/- 0.075 | 0.889 +/- 0.073 |
| supervised | 0.979 +/- 0.028 | 0.979 +/- 0.028 | 0.936 +/- 0.091 | 0.929 +/- 0.062 | 0.929 +/- 0.062 |

## Pooled out-of-fold, with bootstrap intervals

| arm | ROC AUC | PR-AUC |
|---|---|---|
| permuted_control | 0.960 [0.943, 0.976] | 0.947 [0.912, 0.975] |
| semi_supervised | 0.947 [0.925, 0.966] | 0.925 [0.885, 0.960] |
| semi_supervised_confident | 0.932 [0.908, 0.954] | 0.909 [0.863, 0.949] |
| supervised | 0.966 [0.950, 0.981] | 0.964 [0.940, 0.981] |

## Paired differences across the shared folds

| comparison | metric | difference | 95% CI | p |
|---|---|---|---|---|
| supervised vs semi_supervised | roc_auc | +0.015 | [+0.001, +0.029] | 0.037 |
| supervised vs semi_supervised_confident | roc_auc | +0.024 | [+0.011, +0.036] | 0.000 |
| supervised vs permuted_control | roc_auc | +0.010 | [-0.002, +0.022] | 0.099 |
| semi_supervised vs semi_supervised_confident | roc_auc | +0.009 | [-0.005, +0.023] | 0.203 |
| semi_supervised vs permuted_control | roc_auc | -0.004 | [-0.018, +0.009] | 0.543 |
| semi_supervised_confident vs permuted_control | roc_auc | -0.014 | [-0.022, -0.006] | 0.000 |
| supervised vs semi_supervised | recall_positive | -0.004 | [-0.032, +0.028] | 0.882 |
| supervised vs semi_supervised_confident | recall_positive | +0.028 | [-0.024, +0.084] | 0.269 |
| supervised vs permuted_control | recall_positive | +0.020 | [-0.036, +0.088] | 0.550 |
| semi_supervised vs semi_supervised_confident | recall_positive | +0.032 | [-0.016, +0.080] | 0.199 |
| semi_supervised vs permuted_control | recall_positive | +0.024 | [-0.028, +0.092] | 0.526 |
| semi_supervised_confident vs permuted_control | recall_positive | -0.008 | [-0.064, +0.048] | 0.729 |

## Against the `corrected` run, paired over the 25 shared folds

| arm | this run | corrected | difference | 95% CI | p |
|---|---|---|---|---|---|
| permuted_control | 0.968 | 0.952 | +0.016 | [-0.001, +0.037] | 0.062 |
| semi_supervised | 0.964 | 0.958 | +0.006 | [-0.013, +0.022] | 0.514 |
| semi_supervised_confident | 0.955 | 0.945 | +0.009 | [-0.011, +0.032] | 0.417 |
| supervised | 0.979 | 0.966 | +0.013 | [+0.003, +0.024] | 0.006 |

## Clustering method chosen, fold by fold

- `Agglomerative(ward)`: 16 fold(s)
- `KMeans`: 7 fold(s)
- `GMM`: 2 fold(s)

ARI on the training labels: 0.205 ± 0.179
