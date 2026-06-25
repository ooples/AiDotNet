---
title: "RobustnessResult<T>"
description: "Results from robustness analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Results from robustness analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `BaselineScore` | Baseline score without perturbations. |
| `DropoutRobustness` | Performance at each dropout rate (dropout rate → score). |
| `FeatureImportance` | Feature importance via permutation (feature index → importance). |
| `MetricName` | Name of the metric tracked. |
| `NoiseDegradation` | Performance degradation at each noise level. |
| `NoiseRobustness` | Performance at each noise level (noise level → score). |
| `NumFeatures` | Number of features in the dataset. |
| `NumSamples` | Number of samples tested. |
| `OverallRobustnessScore` | Overall robustness score (0 to 1, higher is better). |

