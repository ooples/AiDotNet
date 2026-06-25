---
title: "FairnessConstraint"
description: "Specifies the fairness constraint or metric for model evaluation."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Enums`

Specifies the fairness constraint or metric for model evaluation.

## For Beginners

Imagine a loan approval model. "Fair" could mean:

- Same approval rate across groups (demographic parity)
- Same accuracy across groups (equalized odds)
- Same outcomes for similar people (individual fairness)

These definitions often conflict - you can't optimize for all simultaneously.
Choose the one that aligns with your ethical and legal requirements.

## How It Works

Fairness metrics measure whether a model treats different groups equitably.
Different definitions of "fair" may conflict - choose based on your context and values.

## Fields

| Field | Summary |
|:-----|:--------|
| `AverageOddsDifference` | Average odds difference: Average of TPR and FPR differences. |
| `BalanceNegative` | Balance for negative class: Equal TNR across groups. |
| `BalancePositive` | Balance for positive class: Equal TPR across groups (same as equal opportunity). |
| `BetweenGroupEntropy` | Between-group generalized entropy: Measures fairness across groups. |
| `Calibration` | Calibration: Equal probability calibration across groups. |
| `ConditionalDemographicParity` | Conditional demographic parity: Equal prediction rates within strata. |
| `CounterfactualFairness` | Counterfactual fairness: Same prediction if group membership changed. |
| `DemographicParity` | Demographic parity: Equal positive prediction rates across groups. |
| `DisparateImpact` | Disparate impact ratio: Ratio of positive rates between groups. |
| `EqualOpportunity` | Equal opportunity: Equal TPR (true positive rate) across groups. |
| `EqualizedOdds` | Equalized odds: Equal TPR and FPR across groups. |
| `IndividualFairness` | Individual fairness: Similar individuals receive similar predictions. |
| `None` | No fairness constraint applied. |
| `PredictiveParity` | Predictive parity: Equal PPV (precision) across groups. |
| `StatisticalParityDifference` | Statistical parity difference: Difference in positive prediction rates. |
| `TheilIndex` | Theil index: Inequality measure from economics applied to predictions. |
| `TreatmentEquality` | Treatment equality: Equal ratio of FN to FP across groups. |

