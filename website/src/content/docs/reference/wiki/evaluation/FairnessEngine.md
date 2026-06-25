---
title: "FairnessEngine<T>"
description: "Engine for analyzing fairness and bias in machine learning models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Engine for analyzing fairness and bias in machine learning models.

## For Beginners

Fairness analysis checks if your model treats different groups equally:

- Does the model perform equally well for all demographic groups?
- Are error rates similar across protected groups?
- Are predictions independent of sensitive attributes?

## How It Works

**Key fairness concepts:**

- **Demographic Parity:** Predictions should be independent of group membership
- **Equalized Odds:** TPR and FPR should be equal across groups
- **Equal Opportunity:** TPR should be equal across groups
- **Calibration:** Predicted probabilities should mean the same across groups

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FairnessEngine(FairnessOptions)` | Initializes the fairness engine. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Analyze([],[],Int32[])` | Analyzes fairness of model predictions across groups. |

