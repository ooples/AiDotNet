---
title: "ValidationCurveOptions"
description: "Configuration options for validation curve analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for validation curve analysis.

## For Beginners

A validation curve plots model performance (y-axis) against a
hyperparameter value (x-axis). For example, plotting accuracy vs. regularization strength
helps you find the best regularization value. The point where train and validation scores
are both high and close together is often optimal.

## How It Works

Validation curves show how model performance changes as a hyperparameter varies.
This helps find optimal hyperparameter values and diagnose overfitting.

## Properties

| Property | Summary |
|:-----|:--------|
| `CVFolds` | Number of CV folds. |
| `CVStrategy` | Cross-validation strategy. |
| `ComputeConfidenceIntervals` | Whether to compute confidence intervals. |
| `ConfidenceLevel` | Confidence level. |
| `FindOptimalValue` | Whether to find optimal parameter value. |
| `HigherValueMoreComplex` | Whether higher parameter values mean more model complexity. |
| `MaxDegreeOfParallelism` | Maximum parallelism. |
| `MetricsToTrack` | Metrics to track. |
| `OptimalValueMethod` | How to select optimal value. |
| `ParallelExecution` | Whether to run in parallel. |
| `ParameterName` | Name of the parameter to vary. |
| `ParameterSetterMethod` | Custom parameter setter function name. |
| `ParameterValues` | Values to test for the parameter. |
| `RandomSeed` | Random seed for reproducibility. |
| `Shuffle` | Whether to shuffle before CV. |
| `UseLogScale` | Whether to use logarithmic spacing for parameter values. |

