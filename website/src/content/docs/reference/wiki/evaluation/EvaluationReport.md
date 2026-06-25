---
title: "EvaluationReport<T>"
description: "Comprehensive evaluation report containing all computed metrics and analysis results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Results.Core`

Comprehensive evaluation report containing all computed metrics and analysis results.

## For Beginners

After evaluating your model, this report tells you everything:
how accurate it is, where it struggles, what might be wrong, and how to improve it.
Start with the ExecutiveSummary for a quick overview, then dive into specific sections
as needed.

## How It Works

This is the main result object returned by model evaluation. It contains:

- All computed metrics with confidence intervals
- Dataset statistics
- Diagnostic results (residual analysis, calibration, etc.)
- Warnings and recommendations

## Properties

| Property | Summary |
|:-----|:--------|
| `CalibrationResults` | Calibration analysis results. |
| `ClassificationResults` | Classification-specific results (confusion matrix, ROC, etc.). |
| `ComponentTimings` | Time breakdown by component. |
| `DatasetName` | Dataset name or identifier used for evaluation. |
| `DatasetStatistics` | Statistics about the evaluation dataset. |
| `EvaluationDuration` | Total time taken for evaluation. |
| `EvaluationId` | Unique identifier for this evaluation run. |
| `FairnessResults` | Fairness analysis results. |
| `Metadata` | Metadata about the evaluation configuration. |
| `Metrics` | All computed metrics organized by category. |
| `ModelName` | Model name or identifier being evaluated. |
| `ModelVersion` | Model version, if available. |
| `PrimaryMetric` | Primary (most important) metric for this task. |
| `PrimaryMetricName` | Name of the primary metric. |
| `Recommendations` | Recommendations for improvement. |
| `RegressionResults` | Regression-specific results (residual analysis, etc.). |
| `RobustnessResults` | Robustness analysis results. |
| `TaskType` | Task type (Classification, Regression, etc.). |
| `Timestamp` | Timestamp when evaluation was performed. |
| `UncertaintyResults` | Uncertainty analysis results. |
| `Warnings` | Warnings generated during evaluation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExecutiveSummary` | Generates an executive summary of the evaluation results. |
| `GetMetric(String)` | Gets a specific metric by name. |
| `GetMetricsAsDictionary` | Gets all metrics as a dictionary. |
| `GetQualityAssessment` | Gets the overall quality assessment (Poor, Fair, Good, Excellent). |
| `HasMetric(String)` | Checks if a specific metric was computed. |

