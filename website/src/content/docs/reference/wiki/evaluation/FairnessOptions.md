---
title: "FairnessOptions"
description: "Configuration options for fairness evaluation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for fairness evaluation.

## For Beginners

Fairness metrics check if your model discriminates against certain
groups. For example, in loan approval:

- Does the model approve men and women at similar rates? (Demographic Parity)
- Is the model equally accurate for all races? (Equalized Odds)
- When it predicts "will repay", is it equally reliable for all groups? (Calibration)

Note: Different fairness metrics often conflict - you can't optimize for all simultaneously.

## How It Works

Fairness evaluation checks whether a model treats different demographic groups equitably.
This is critical for applications in lending, hiring, criminal justice, healthcare, etc.

## Properties

| Property | Summary |
|:-----|:--------|
| `BootstrapSamples` | Number of bootstrap samples. |
| `ComputeConditionalFairness` | Whether to compute conditional fairness. |
| `ComputeConfidenceIntervals` | Whether to compute confidence intervals. |
| `ComputeCounterfactualFairness` | Whether to compute counterfactual fairness (requires causal model). |
| `ComputeIntersectionalFairness` | Whether to compute intersectional fairness. |
| `ComputeTheilDecomposition` | Whether to generate Theil index decomposition. |
| `ConfidenceLevel` | Confidence level. |
| `DisparityThreshold` | Threshold for flagging disparity. |
| `FlagUnfairPredictions` | Whether to flag unfair predictions. |
| `IncludeRecommendations` | Whether to include recommendations. |
| `IndividualUnfairnessThreshold` | Threshold for individual unfairness. |
| `LegitimateFeatureIndices` | Legitimate feature indices for conditional fairness. |
| `MaxIntersectionSize` | Maximum intersection size. |
| `MetricsToCompute` | Fairness constraints to evaluate. |
| `MinGroupSize` | Minimum group size for reliable metrics. |
| `PerformSignificanceTests` | Whether to perform significance tests for disparities. |
| `PrivilegedGroupValues` | Reference/privileged group values for each protected attribute. |
| `ProtectedAttributeIndices` | Protected attribute column indices. |
| `ProtectedAttributeNames` | Protected attribute names for reporting. |
| `ReportFormat` | Report format. |
| `SignificanceLevel` | Significance level for tests. |

