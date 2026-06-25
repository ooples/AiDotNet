---
title: "MeanAbsoluteErrorFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Mean Absolute Error (MAE) to evaluate model performance, particularly for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Mean Absolute Error (MAE) to evaluate model performance, particularly for regression tasks.

## For Beginners

This calculator helps evaluate how well your model is performing on regression tasks
(where you're predicting continuous values like prices, temperatures, etc.) by measuring the average
size of the errors in your predictions without considering their direction.

Mean Absolute Error (MAE) is one of the simplest and most intuitive ways to measure prediction errors:

- It calculates the absolute difference between each prediction and the actual value
- It then takes the average of all these differences
- The result tells you, on average, how far off your predictions are

How MAE works:

- Take the difference between each predicted value and actual value
- Convert all differences to positive numbers (take the absolute value)
- Calculate the average of these absolute differences

Think of it like this:
Imagine you're predicting house prices:

- For one house, you predict $200,000 but the actual price is $220,000 (error of $20,000)
- For another house, you predict $350,000 but the actual price is $330,000 (error of $20,000)
- The MAE would be $20,000, telling you that on average, your predictions are off by $20,000

Key characteristics of MAE:

- It treats all errors equally (unlike Mean Squared Error which penalizes large errors more)
- It's measured in the same units as your original data (dollars, degrees, etc.)
- It's less sensitive to outliers than Mean Squared Error
- Lower values are better (0 would be perfect predictions)

Common applications include:

- Price prediction
- Temperature forecasting
- Any regression task where you want errors measured in the original units
- Situations where outliers should not have outsized influence

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeanAbsoluteErrorFitnessCalculator(DataSetType)` | Initializes a new instance of the MeanAbsoluteErrorFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Mean Absolute Error (MAE) fitness score for the given dataset. |

