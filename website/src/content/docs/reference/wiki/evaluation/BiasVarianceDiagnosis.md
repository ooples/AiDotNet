---
title: "BiasVarianceDiagnosis"
description: "Specifies the diagnosed bias-variance condition of a model."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Enums`

Specifies the diagnosed bias-variance condition of a model.

## For Beginners

Think of it like throwing darts:

- **High bias:** Darts consistently miss the bullseye in the same direction (systematic error)
- **High variance:** Darts scattered all over (inconsistent)
- **Good fit:** Darts clustered around the bullseye

## How It Works

Bias-variance tradeoff is fundamental to understanding model performance:

- **Bias:** Error from overly simplistic assumptions (underfitting)
- **Variance:** Error from sensitivity to training data fluctuations (overfitting)

## Fields

| Field | Summary |
|:-----|:--------|
| `GoodFit` | Model has a good bias-variance balance: Good performance on both training and test data. |
| `HighBias` | Model exhibits high bias (underfitting): Poor performance on both training and test data. |
| `HighBiasHighVariance` | Both high bias and high variance: Rare case indicating fundamental model issues. |
| `HighVariance` | Model exhibits high variance (overfitting): Good training performance but poor test performance. |
| `NeedsMoreData` | Model performance is still improving with more data. |
| `SevereOverfit` | Model is severely overfitting: Perfect training but random test performance. |
| `SevereUnderfit` | Model is severely underfitting: Training error is extremely high. |
| `Undetermined` | Unable to determine diagnosis: Insufficient data or ambiguous results. |
| `Unknown` | Unable to determine diagnosis: Alias for Undetermined. |

