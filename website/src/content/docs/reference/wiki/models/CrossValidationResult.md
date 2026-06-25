---
title: "CrossValidationResult<T, TInput, TOutput>"
description: "Aggregates results from all folds in a cross-validation procedure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Aggregates results from all folds in a cross-validation procedure.

## For Beginners

Cross-validation helps you understand how well your model will perform
on new data by testing it on several different train/test splits. This class combines
the results from all those tests to give you an overall picture of your model's performance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CrossValidationResult(List<FoldResult<,,>>,TimeSpan)` | Creates a new instance of the CrossValidationResult class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjustedRandIndexStats` | Gets basic statistics for Adjusted Rand Index values across folds, or null if not applicable. |
| `AverageTrainingTime` | Gets the average time taken to train the model across all folds. |
| `CalinskiHarabaszIndexStats` | Gets basic statistics for Calinski-Harabasz Index values across folds, or null if not applicable. |
| `DaviesBouldinIndexStats` | Gets basic statistics for Davies-Bouldin Index values across folds, or null if not applicable. |
| `FeatureImportanceStats` | Gets a dictionary of feature importance scores aggregated across all folds. |
| `FoldCount` | Gets the number of folds used in cross-validation. |
| `FoldResults` | Gets the individual results for each fold. |
| `MAEStats` | Gets basic statistics for MAE values across folds. |
| `R2Stats` | Gets basic statistics (mean, standard deviation, etc.) for R² values across folds. |
| `RMSEStats` | Gets basic statistics for RMSE values across folds. |
| `SilhouetteScoreStats` | Gets basic statistics for Silhouette Score values across folds, or null if not applicable. |
| `TotalTime` | Gets the total time taken for the entire cross-validation process. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateFeatureImportance(List<FoldResult<,,>>)` | Combines feature importance scores from all folds and calculates statistics. |
| `ExtractClusteringMetricValues(List<FoldResult<,,>>,Func<ClusteringMetrics<>,>)` | Extracts non-null clustering metric values from fold results using a selector function. |
| `GenerateReport` | Generates a comprehensive summary report of the cross-validation results. |
| `GetMetricStats(MetricType)` | Gets summary statistics for a specific metric across all folds. |

