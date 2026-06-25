---
title: "MeanSquaredErrorFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Mean Squared Error (MSE) to evaluate model performance, particularly for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Mean Squared Error (MSE) to evaluate model performance, particularly for regression tasks.

## For Beginners

This calculator helps evaluate how well your model is performing on regression tasks
(where you're predicting continuous values like prices, temperatures, etc.) by measuring the average
of the squared differences between predictions and actual values.

Mean Squared Error (MSE) is one of the most commonly used metrics in machine learning:

- It calculates the square of the difference between each prediction and the actual value
- It then takes the average of all these squared differences
- The result tells you how large your errors are, with larger errors being penalized more heavily

How MSE works:

- Take the difference between each predicted value and actual value
- Square each difference (multiply it by itself)
- Calculate the average of these squared differences

Think of it like this:
Imagine you're predicting house prices:

- For one house, you predict $200,000 but the actual price is $220,000 (error of $20,000)
- For another house, you predict $350,000 but the actual price is $330,000 (error of $20,000)
- When squared, both errors become 400,000,000
- The MSE would be 400,000,000, which is much larger than the original error

Key characteristics of MSE:

- It penalizes larger errors more heavily than smaller ones (due to squaring)
- It's measured in squared units (e.g., squared dollars, squared degrees)
- It's more sensitive to outliers than Mean Absolute Error
- Lower values are better (0 would be perfect predictions)

Common applications include:

- Training many types of regression models
- Situations where large errors should be penalized more heavily
- When you want a differentiable loss function for optimization

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeanSquaredErrorFitnessCalculator(DataSetType)` | Initializes a new instance of the MeanSquaredErrorFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Mean Squared Error (MSE) fitness score for the given dataset. |

