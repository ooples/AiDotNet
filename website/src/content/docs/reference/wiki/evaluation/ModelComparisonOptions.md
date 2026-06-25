---
title: "ModelComparisonOptions"
description: "Configuration options for comparing multiple models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for comparing multiple models.

## For Beginners

When comparing models, you might find Model A has 85% accuracy and
Model B has 87% accuracy. But is that 2% difference real or just noise? Statistical tests
help answer this by computing p-values (probability the difference is due to chance).

## How It Works

Model comparison provides statistical tests to determine if one model is significantly
better than another, or to rank multiple models.

## Properties

| Property | Summary |
|:-----|:--------|
| `ApplyMultipleTestingCorrection` | Whether to apply multiple testing correction. |
| `BootstrapSamples` | Number of bootstrap samples. |
| `CVFolds` | Number of CV folds. |
| `CVRepeats` | Number of CV repetitions. |
| `CVStrategy` | Cross-validation strategy for comparison. |
| `ComputeConfidenceIntervals` | Whether to compute confidence intervals for differences. |
| `ComputeEffectSizes` | Whether to compute effect sizes. |
| `ConfidenceLevel` | Confidence level for intervals. |
| `CorrectionMethod` | Multiple testing correction method. |
| `EffectSizeMeasure` | Effect size measure to use. |
| `GenerateCriticalDifferenceDiagram` | Whether to create critical difference diagram data. |
| `GenerateRanking` | Whether to generate ranking of models. |
| `MultipleComparisonTest` | Test for comparing multiple models simultaneously. |
| `PairwiseTest` | Statistical test for pairwise comparison. |
| `ParallelExecution` | Whether to run comparisons in parallel. |
| `PostHocTest` | Post-hoc test after significant Friedman test. |
| `PrimaryMetric` | Primary metric for comparison. |
| `RandomSeed` | Random seed for reproducibility. |
| `RopeWidth` | ROPE (Region of Practical Equivalence) for Bayesian comparison. |
| `SignificanceLevel` | Significance level (alpha) for hypothesis tests. |
| `UseBayesianComparison` | Whether to use Bayesian comparison. |

