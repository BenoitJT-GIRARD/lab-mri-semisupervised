# corrected-20260904-003804

Protocol: **corrected**

- dataset fingerprint: `8f22a69ffa46ca1f`
- evaluation images: 99
- folds: 25 (5 splits x 5 repeats)
- unlabelled pool: 1311

## Per-arm means across folds

| arm | roc_auc | pr_auc | recall_positive | f1_macro | accuracy |
|---|---|---|---|---|---|
| joint_permuted_control | 0.811 +/- 0.142 | 0.839 +/- 0.127 | 0.752 +/- 0.216 | 0.705 +/- 0.133 | 0.721 +/- 0.116 |
| self_training | 0.922 +/- 0.066 | 0.931 +/- 0.060 | 0.844 +/- 0.150 | 0.842 +/- 0.094 | 0.846 +/- 0.088 |
| self_training_control | 0.931 +/- 0.067 | 0.940 +/- 0.063 | 0.860 +/- 0.126 | 0.834 +/- 0.083 | 0.836 +/- 0.082 |
| semi_supervised_joint | 0.925 +/- 0.064 | 0.934 +/- 0.059 | 0.868 +/- 0.141 | 0.830 +/- 0.091 | 0.834 +/- 0.087 |

## Pooled out-of-fold, with bootstrap intervals

| arm | ROC AUC | PR-AUC |
|---|---|---|
| joint_permuted_control | 0.736 [0.691, 0.778] | 0.744 [0.683, 0.800] |
| self_training | 0.847 [0.812, 0.881] | 0.839 [0.790, 0.880] |
| self_training_control | 0.773 [0.731, 0.814] | 0.803 [0.753, 0.847] |
| semi_supervised_joint | 0.886 [0.856, 0.915] | 0.895 [0.860, 0.928] |

## Paired differences across the shared folds

| comparison | metric | difference | 95% CI | p |
|---|---|---|---|---|
| semi_supervised_joint vs joint_permuted_control | roc_auc | +0.113 | [+0.050, +0.180] | 0.002 |
| semi_supervised_joint vs self_training | roc_auc | +0.003 | [-0.030, +0.036] | 0.898 |
| semi_supervised_joint vs self_training_control | roc_auc | -0.006 | [-0.035, +0.018] | 0.612 |
| joint_permuted_control vs self_training | roc_auc | -0.111 | [-0.170, -0.051] | 0.000 |
| joint_permuted_control vs self_training_control | roc_auc | -0.120 | [-0.177, -0.065] | 0.000 |
| self_training vs self_training_control | roc_auc | -0.009 | [-0.040, +0.019] | 0.530 |
| semi_supervised_joint vs joint_permuted_control | recall_positive | +0.116 | [+0.028, +0.204] | 0.012 |
| semi_supervised_joint vs self_training | recall_positive | +0.024 | [-0.032, +0.088] | 0.502 |
| semi_supervised_joint vs self_training_control | recall_positive | +0.008 | [-0.052, +0.060] | 0.768 |
| joint_permuted_control vs self_training | recall_positive | -0.092 | [-0.168, -0.020] | 0.008 |
| joint_permuted_control vs self_training_control | recall_positive | -0.108 | [-0.192, -0.024] | 0.016 |
| self_training vs self_training_control | recall_positive | -0.016 | [-0.084, +0.044] | 0.692 |

## Clustering method chosen, fold by fold

- `Agglomerative(ward)`: 24 fold(s)
- `KMeans`: 1 fold(s)

ARI on the training labels: 0.410 ± 0.177
