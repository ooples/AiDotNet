---
title: "CrossValidationStrategy"
description: "Specifies the cross-validation strategy to use for model evaluation."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Enums`

Specifies the cross-validation strategy to use for model evaluation.

## For Beginners

Cross-validation is like giving your model multiple "practice tests" on
different portions of your data. Each strategy splits the data differently:

- **KFold**: Divides data into K equal parts, trains on K-1, tests on 1, repeats K times
- **Stratified**: Same as KFold but preserves class proportions (important for imbalanced data)
- **TimeSeries**: Respects time order - always trains on past, tests on future
- **Group**: Keeps related samples together (e.g., all data from one patient stays together)

Choose based on your data type and whether you have class imbalance, time dependencies, or grouped samples.

## How It Works

Cross-validation is a technique for assessing how well a model generalizes to independent data.
Different strategies are appropriate for different data types and problem domains.

## Fields

| Field | Summary |
|:-----|:--------|
| `AdversarialSplit` | Adversarial Split deliberately creates challenging test sets to evaluate robustness. |
| `BlockedTimeSeriesSplit` | Blocked Time Series Split adds gaps between train and test to prevent information leakage. |
| `BootstrapCV` | Bootstrap CV uses bootstrap resampling (sampling with replacement). |
| `CombinatorialPurgedCV` | Combinatorial Purged CV generates all possible train/test combinations with purging. |
| `GroupKFold` | Group K-Fold ensures samples from the same group never appear in both train and test. |
| `GroupShuffleSplit` | Group Shuffle-Split randomly samples groups for train/test while respecting group boundaries. |
| `KFold` | Standard K-Fold cross-validation. |
| `LeaveOneOut` | Leave-One-Out (LOO) uses each sample as a single test case while training on all others. |
| `LeavePOut` | Leave-P-Out uses all possible combinations of P samples as test sets. |
| `MonteCarloCV` | Monte Carlo CV (repeated random sub-sampling) creates random train/test splits. |
| `PurgedKFold` | Purged K-Fold removes samples from training that are temporally close to test samples. |
| `RepeatedKFold` | Repeated K-Fold runs standard K-Fold multiple times with different random shuffles. |
| `RepeatedStratifiedKFold` | Repeated Stratified K-Fold combines stratification with repetition. |
| `ShuffleSplit` | Shuffle-Split creates random train/test splits with configurable sizes. |
| `SlidingWindowSplit` | Sliding Window Split uses fixed-size training windows that slide through time. |
| `StratifiedKFold` | Stratified K-Fold preserves the percentage of samples for each class in each fold. |
| `StratifiedShuffleSplit` | Stratified Shuffle-Split maintains class proportions in random splits. |
| `TimeSeriesSplit` | Time Series Split uses expanding training windows. |

