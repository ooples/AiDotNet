---
title: "LogCoshLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Log-Cosh Loss to evaluate model performance, particularly for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Log-Cosh Loss to evaluate model performance, particularly for regression tasks.

## For Beginners

This calculator helps evaluate how well your model is performing on regression tasks
(where you're predicting continuous values like prices, temperatures, etc.) while being less
sensitive to outliers than some other methods.

Log-Cosh Loss is a smooth approximation that combines the best features of two common loss functions:

- Mean Squared Error (MSE): Good for most predictions but very sensitive to outliers
- Mean Absolute Error (MAE): Less sensitive to outliers but has mathematical limitations

How Log-Cosh Loss works:

- For small errors: It behaves almost like Mean Squared Error
- For large errors: It behaves more like Mean Absolute Error
- It uses the natural logarithm of the hyperbolic cosine function to achieve this balance

Think of it like this:
Imagine you're measuring how far off your predictions are:

- For small mistakes, Log-Cosh Loss increases quickly (like MSE)
- For large mistakes, it increases more slowly (like MAE)
- But unlike MAE, it's smooth everywhere (which helps with optimization)

Common applications include:

- Price prediction
- Any regression task where outliers might be present
- Situations where you want a loss function that's both smooth and robust

The main advantage of Log-Cosh Loss is that it's less sensitive to outliers than MSE
but still has nice mathematical properties that make it easier to optimize than MAE.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LogCoshLossFitnessCalculator(DataSetType)` | Initializes a new instance of the LogCoshLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Log-Cosh Loss fitness score for the given dataset. |

