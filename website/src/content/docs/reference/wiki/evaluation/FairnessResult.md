---
title: "FairnessResult<T>"
description: "Results from fairness analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Results from fairness analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageOddsDifference` | Average Odds Difference: mean of TPR and FPR differences. |
| `DemographicParityDifference` | Demographic Parity Difference: max - min positive prediction rate across groups. |
| `DisparateImpactRatio` | Disparate Impact Ratio: min positive rate / max positive rate. |
| `EqualOpportunityDifference` | Equal Opportunity Difference: max - min TPR across groups. |
| `EqualizedOddsDifference` | Equalized Odds Difference: max of TPR and FPR differences across groups. |
| `GroupMetrics` | Per-group fairness metrics. |
| `Groups` | Array of group identifiers. |
| `IsFair` | Whether the model is considered fair based on configured thresholds. |
| `NumSamples` | Total number of samples analyzed. |
| `TheilIndex` | Theil Index: measures inequality in model benefit distribution. |

