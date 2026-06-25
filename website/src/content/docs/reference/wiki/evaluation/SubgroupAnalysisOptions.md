---
title: "SubgroupAnalysisOptions"
description: "Configuration options for subgroup (slice-based) analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for subgroup (slice-based) analysis.

## For Beginners

Your model might have 90% accuracy overall, but:

- 95% accuracy on common cases, 50% on rare cases
- 92% on young users, 75% on elderly users
- 98% on clean data, 60% on noisy data

Subgroup analysis reveals these hidden variations that overall metrics miss.

## How It Works

Subgroup analysis computes metrics for different slices of your data, helping identify
where the model performs well or poorly. This is essential for understanding model behavior.

## Properties

| Property | Summary |
|:-----|:--------|
| `ApplyMultipleTestingCorrection` | Whether to apply multiple testing correction. |
| `AutoDetectCategoricalFeatures` | Whether to auto-detect categorical features for slicing. |
| `BinContinuousFeatures` | Whether to bin continuous features for slicing. |
| `BinningStrategy` | Binning strategy for continuous features. |
| `ComputeConfidenceIntervals` | Whether to compute confidence intervals per slice. |
| `ComputeSliceImportance` | Whether to compute slice importance (which slices matter most). |
| `ComputeSliceIntersections` | Whether to compute intersections of slices. |
| `ConfidenceLevel` | Confidence level. |
| `ContinuousBins` | Number of bins for continuous features. |
| `CustomBinEdges` | Custom bin edges for specific features. |
| `GenerateSliceErrorAnalysis` | Whether to generate error analysis per slice. |
| `IdentifyUnderperformingSlices` | Whether to identify underperforming slices. |
| `IncludeDistributionStats` | Whether to include slice distribution statistics. |
| `IncludeRecommendations` | Whether to include recommendations per slice. |
| `MaxCategoricalUniques` | Maximum unique values for auto-detected categorical features. |
| `MaxIntersectionSize` | Maximum intersection size. |
| `MaxSlicesInReport` | Maximum slices to include in report. |
| `MetricsToCompute` | Metrics to compute per slice. |
| `MinSliceSize` | Minimum samples per slice for reliable metrics. |
| `ParallelExecution` | Whether to run analyses in parallel. |
| `ReportFormat` | Report format. |
| `SignificanceLevel` | Significance level for difference tests. |
| `SliceFeatureIndices` | Feature indices to slice by. |
| `SliceFeatureNames` | Feature names for slicing (alternative to indices). |
| `SliceSortOrder` | Sort order for slices in report. |
| `TestSignificantDifference` | Whether to perform statistical test vs overall. |
| `UnderperformanceThreshold` | Threshold for underperformance (relative to overall). |

