---
title: "CrossValidationOptions"
description: "Configuration options for cross-validation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for cross-validation.

## For Beginners

Cross-validation helps you understand how well your model will
perform on new, unseen data. Instead of a single train/test split, it creates multiple
splits and averages the results for a more reliable estimate.

## How It Works

Cross-validation provides robust performance estimates by training and testing on
different subsets of data. These options control the validation strategy.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComputeFoldVariance` | Whether to compute per-fold variance. |
| `EmbargoPeriod` | Embargo period for combinatorial purged CV. |
| `Gap` | Gap between train and test sets for time series CV. |
| `GroupColumnIndex` | Group column/feature index for group-aware CV. |
| `GroupLabels` | Group labels array for group-aware CV. |
| `InnerCVOptions` | Inner CV options for nested cross-validation. |
| `LeavePOutSamples` | Number of samples to leave out for leave-P-out CV. |
| `MaxDegreeOfParallelism` | Maximum degree of parallelism. |
| `MaxTrainSize` | Maximum training size for time series CV. |
| `MinTrainSize` | Minimum training size required. |
| `NumberOfFolds` | Number of folds (K). |
| `NumberOfRepeats` | Number of repetitions for repeated CV. |
| `ParallelExecution` | Whether to run folds in parallel. |
| `PurgePeriod` | Purge period for purged K-fold (financial CV). |
| `RandomSeed` | Random seed for reproducibility. |
| `ReturnPredictions` | Whether to return predictions from each fold. |
| `ReturnTrainedModels` | Whether to return trained models from each fold. |
| `Shuffle` | Whether to shuffle data before splitting. |
| `StepSize` | Step size for sliding window CV. |
| `Strategy` | Cross-validation strategy to use. |
| `TestSize` | Test size ratio for shuffle split strategies. |
| `UseOutOfFoldPredictions` | Whether to use out-of-fold predictions for evaluation. |
| `WindowSize` | Window size for sliding window CV. |

