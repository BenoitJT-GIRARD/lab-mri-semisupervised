# corrected-20260904-010836

Protocol: **corrected**

- dataset fingerprint: `8f22a69ffa46ca1f`
- evaluation images: 99
- folds: 25 (5 splits x 5 repeats)
- unlabelled pool: 1311

## Per-arm means across folds

| arm | roc_auc | pr_auc | recall_positive | f1_macro | accuracy |
|---|---|---|---|---|---|
| joint_permuted_control | 0.922 +/- 0.063 | 0.930 +/- 0.065 | 0.860 +/- 0.144 | 0.837 +/- 0.100 | 0.840 +/- 0.097 |
| self_training | 0.963 +/- 0.036 | 0.962 +/- 0.039 | 0.916 +/- 0.121 | 0.895 +/- 0.075 | 0.897 +/- 0.072 |
| self_training_control | 0.948 +/- 0.069 | 0.952 +/- 0.063 | 0.884 +/- 0.131 | 0.856 +/- 0.108 | 0.860 +/- 0.102 |
| semi_supervised_joint | 0.955 +/- 0.041 | 0.955 +/- 0.046 | 0.908 +/- 0.108 | 0.883 +/- 0.054 | 0.885 +/- 0.053 |

## Pooled out-of-fold, with bootstrap intervals

| arm | ROC AUC | PR-AUC |
|---|---|---|
| joint_permuted_control | 0.866 [0.833, 0.896] | 0.872 [0.831, 0.909] |
| self_training | 0.938 [0.915, 0.956] | 0.940 [0.914, 0.961] |
| self_training_control | 0.888 [0.856, 0.917] | 0.905 [0.872, 0.934] |
| semi_supervised_joint | 0.937 [0.914, 0.958] | 0.935 [0.905, 0.960] |

## Paired differences across the shared folds

| comparison | metric | difference | 95% CI | p |
|---|---|---|---|---|
| semi_supervised_joint vs joint_permuted_control | roc_auc | +0.033 | [+0.009, +0.057] | 0.008 |
| semi_supervised_joint vs self_training | roc_auc | -0.008 | [-0.022, +0.006] | 0.296 |
| semi_supervised_joint vs self_training_control | roc_auc | +0.007 | [-0.010, +0.029] | 0.557 |
| joint_permuted_control vs self_training | roc_auc | -0.041 | [-0.064, -0.017] | 0.001 |
| joint_permuted_control vs self_training_control | roc_auc | -0.026 | [-0.060, +0.012] | 0.158 |
| self_training vs self_training_control | roc_auc | +0.015 | [-0.008, +0.043] | 0.256 |
| semi_supervised_joint vs joint_permuted_control | recall_positive | +0.048 | [-0.016, +0.108] | 0.134 |
| semi_supervised_joint vs self_training | recall_positive | -0.008 | [-0.044, +0.028] | 0.703 |
| semi_supervised_joint vs self_training_control | recall_positive | +0.024 | [-0.020, +0.072] | 0.278 |
| joint_permuted_control vs self_training | recall_positive | -0.056 | [-0.108, -0.004] | 0.026 |
| joint_permuted_control vs self_training_control | recall_positive | -0.024 | [-0.080, +0.032] | 0.366 |
| self_training vs self_training_control | recall_positive | +0.032 | [-0.004, +0.076] | 0.076 |

## Clustering method chosen, fold by fold

- `Agglomerative(ward)`: 24 fold(s)
- `GMM`: 1 fold(s)

ARI on the training labels: 0.457 ± 0.096
