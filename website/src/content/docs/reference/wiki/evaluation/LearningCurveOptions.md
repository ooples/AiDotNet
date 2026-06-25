---
title: "LearningCurveOptions"
description: "Configuration options for learning curve analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for learning curve analysis.

## For Beginners

A learning curve plots model performance (y-axis) against training
set size (x-axis). It helps answer questions like:

- Will collecting more data help? (yes if training curve is still improving)
- Is the model overfitting? (yes if train score >> validation score)
- Is the model underfitting? (yes if both scores are low and flat)

## How It Works

Learning curves show how model performance changes as training data size increases.
This helps diagnose bias-variance tradeoffs and determine if more data would help.

## Properties

| Property | Summary |
|:-----|:--------|
| `BiasVarianceBootstrapSamples` | Number of bootstrap samples for bias-variance decomposition. |
| `CVFolds` | Number of CV folds. |
| `CVStrategy` | Cross-validation strategy for each point. |
| `ComputeBiasVarianceDecomposition` | Whether to compute bias-variance decomposition. |
| `ComputeConfidenceIntervals` | Whether to compute confidence intervals at each point. |
| `ConfidenceLevel` | Confidence level for intervals. |
| `DiagnoseBiasVariance` | Whether to diagnose bias-variance condition. |
| `ExtrapolateCurve` | Whether to extrapolate learning curve beyond available data. |
| `ExtrapolationTarget` | Extrapolation target size (ratio or absolute). |
| `HighBiasThreshold` | Threshold for high bias diagnosis (low scores). |
| `HighVarianceThreshold` | Threshold for high variance diagnosis (train-test gap). |
| `MaxDegreeOfParallelism` | Maximum degree of parallelism. |
| `MetricsToTrack` | Metrics to track on learning curve. |
| `MinTrainSizeRatio` | Minimum training size ratio. |
| `NumberOfPoints` | Number of training sizes to auto-generate. |
| `ParallelExecution` | Whether to run evaluations in parallel. |
| `RandomSeed` | Random seed for reproducibility. |
| `Shuffle` | Whether to shuffle before splitting. |
| `TrainSizes` | Training set sizes to evaluate. |
| `TrainSizesAreRatios` | Whether TrainSizes are ratios (0-1) or absolute counts. |

