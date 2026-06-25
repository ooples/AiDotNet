---
title: "AdjustedRSquaredFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses the Adjusted R-Squared metric to evaluate model performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses the Adjusted R-Squared metric to evaluate model performance.

## For Beginners

This calculator helps you evaluate how well your model fits the data using 
a metric called "Adjusted R-Squared."

Regular R-Squared (also called the coefficient of determination) measures how well your model 
explains the variation in your data, ranging from 0 to 1:

- 0 means your model doesn't explain any of the variation
- 1 means your model perfectly explains all variation

However, regular R-Squared has a problem: it always increases when you add more features to your model,
even if those features don't actually help with predictions.

Adjusted R-Squared fixes this issue by penalizing models that use too many features. It's like
R-Squared with a built-in protection against overly complex models. This helps you find the
right balance between model complexity and accuracy.

Unlike regular R-Squared, Adjusted R-Squared can be negative if your model performs very poorly.
Generally, you want this value to be as close to 1 as possible.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdjustedRSquaredFitnessCalculator(DataSetType)` | Initializes a new instance of the AdjustedRSquaredFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Retrieves the Adjusted R-Squared value from the dataset statistics. |

