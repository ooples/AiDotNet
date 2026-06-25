---
title: "IOrdinalClassifier<T>"
description: "Interface for ordinal classification (ordinal regression) models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for ordinal classification (ordinal regression) models.

## For Beginners

Ordinal classification handles categories with a natural order,
like ratings (1-5 stars), education levels, or pain severity. Unlike regular classification,
the order matters - predicting 5 stars when true is 4 stars is better than predicting 1 star.

## How It Works

**Key differences from regular classification:**

- Classes have a natural ordering (1 < 2 < 3 < ...)
- Errors near the true class are less severe than distant errors
- Models often use cumulative probabilities P(Y ≤ k)

**Common approaches:**

- **Threshold models:** Learn thresholds that separate ordered classes
- **Proportional odds:** Ordinal logistic regression
- **Regression-based:** Treat ordinal as continuous, round predictions

## Properties

| Property | Summary |
|:-----|:--------|
| `OrderedClasses` | Gets the ordered class labels from lowest to highest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Predict(Matrix<>)` | Predicts ordinal class labels. |
| `PredictCumulativeProbabilities(Matrix<>)` | Predicts cumulative probabilities P(Y ≤ k) for each class threshold. |

