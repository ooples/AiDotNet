---
title: "UnivariateScoringFunction"
description: "Defines the scoring functions available for univariate feature selection."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the scoring functions available for univariate feature selection.

## For Beginners

These are different statistical tests that measure how much a single feature
is related to the target variable you're trying to predict.

## How It Works

Each scoring function is best suited for different types of data:

- Chi-Squared: Best for categorical features and categorical targets
- ANOVA F-Value: Best for continuous features and categorical targets
- Mutual Information: Works well for both categorical and continuous features with any target type

## Fields

| Field | Summary |
|:-----|:--------|
| `ChiSquared` | Uses the Chi-Squared test to measure the dependence between categorical features and the target. |
| `FValue` | Uses ANOVA F-value to measure how well continuous features can distinguish between different classes. |
| `MutualInformation` | Uses Mutual Information to measure how much knowing one feature tells you about the target. |

