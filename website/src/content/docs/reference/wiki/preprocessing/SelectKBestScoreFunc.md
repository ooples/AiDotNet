---
title: "SelectKBestScoreFunc"
description: "Scoring functions for SelectKBest."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Scoring functions for SelectKBest.

## Fields

| Field | Summary |
|:-----|:--------|
| `Chi2` | Chi-squared statistic for classification (features must be non-negative). |
| `FClassif` | ANOVA F-value for classification. |
| `FRegression` | F-value for regression (correlation-based). |
| `MutualInfoRegression` | Mutual information for regression (captures non-linear relationships). |

