---
title: "EvaluationOptions<T>"
description: "Configuration options for model evaluation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for model evaluation.

## For Beginners

These options control how your model is evaluated. By default,
the framework auto-detects what kind of model you have (classification, regression, etc.)
and computes appropriate metrics. You can customize this if needed.

## How It Works

Controls which metrics are computed, confidence interval settings, and output preferences.
All properties are nullable with sensible defaults applied internally.

## Properties

| Property | Summary |
|:-----|:--------|
| `BootstrapSamples` | Number of bootstrap samples for confidence intervals. |
| `ClassificationThreshold` | Custom classification threshold. |
| `ComputeAllMetrics` | Whether to compute all available metrics. |
| `ComputeCalibrationMetrics` | Whether to compute probability calibration metrics. |
| `ComputeConfidenceIntervals` | Whether to compute confidence intervals for metrics. |
| `ComputePermutationImportance` | Whether to compute feature importance via permutation. |
| `ComputeThresholdAnalysis` | Whether to compute threshold analysis for binary classification. |
| `ConfidenceIntervalMethod` | Method for computing confidence intervals. |
| `ConfidenceLevel` | Confidence level for intervals. |
| `EmitWarnings` | Whether to warn about potential issues (class imbalance, etc.). |
| `FalseNegativeCost` | Cost of false negatives for cost-sensitive evaluation. |
| `FalsePositiveCost` | Cost of false positives for cost-sensitive evaluation. |
| `IncludePerClassMetrics` | Whether to include per-class metrics for classification. |
| `MaxDegreeOfParallelism` | Number of parallel threads for computation. |
| `MemoryLimitBytes` | Memory limit in bytes for evaluation. |
| `MetricsToCompute` | Which metrics to compute. |
| `MultiClassAveraging` | Averaging method for multi-class metrics. |
| `PerformInfluenceAnalysis` | Whether to perform influence analysis for regression. |
| `PerformResidualAnalysis` | Whether to perform residual analysis for regression. |
| `PermutationRounds` | Number of permutation rounds for importance. |
| `PositiveClassLabel` | Positive class label for binary classification. |
| `RandomSeed` | Random seed for reproducible confidence intervals. |
| `ThresholdSelectionMethod` | Threshold selection method for binary classification. |
| `TrackComputationTime` | Whether to track computation time for each metric. |

