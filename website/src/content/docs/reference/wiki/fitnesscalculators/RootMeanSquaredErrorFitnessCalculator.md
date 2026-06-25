---
title: "RootMeanSquaredErrorFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Root Mean Squared Error (RMSE) to evaluate model performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Root Mean Squared Error (RMSE) to evaluate model performance.

## For Beginners

This calculator helps evaluate how well your model is performing by measuring
the average size of the errors in your predictions, with a special emphasis on larger errors.

Root Mean Squared Error (RMSE) is one of the most commonly used metrics in machine learning and is:

- A measure of how far off your predictions are from the actual values
- Calculated by taking the square root of the average of squared differences between predictions and actual values
- Expressed in the same units as your target variable (which makes it easy to interpret)

How RMSE works:

1. Calculate the difference between each predicted value and the actual value
2. Square each of these differences (to make all values positive and emphasize larger errors)
3. Calculate the average (mean) of these squared differences
4. Take the square root of this average to get back to the original units

Think of it like this:
Imagine you're predicting house prices in dollars:

- If your RMSE is $50,000, it means your predictions are off by about $50,000 on average
- But because of the squaring step, being off by $100,000 on one house is penalized more than

being off by $50,000 on two houses

Key characteristics of RMSE:

- Lower values are better (0 would be perfect predictions)
- It penalizes larger errors more heavily than smaller ones
- It's sensitive to outliers (a few very bad predictions can significantly increase RMSE)
- It's in the same units as your target variable (making it easy to understand)

When to use RMSE:

- When you want to heavily penalize large errors
- When outliers in your predictions should be considered important
- When you want a metric that's in the same units as your target variable
- For regression problems (predicting continuous values like prices, temperatures, etc.)

RMSE is one of the most popular metrics for regression tasks because it provides a clear,
interpretable measure of prediction error in the original units of the target variable.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RootMeanSquaredErrorFitnessCalculator(DataSetType)` | Initializes a new instance of the RootMeanSquaredErrorFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Root Mean Squared Error (RMSE) fitness score for the given dataset. |

