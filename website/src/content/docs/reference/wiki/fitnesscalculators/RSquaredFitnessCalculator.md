---
title: "RSquaredFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses R-Squared (R²) to evaluate model performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses R-Squared (R²) to evaluate model performance.

## For Beginners

This calculator helps evaluate how well your model is performing by measuring
the proportion of variance in your target variable that is explained by your model.

R-Squared (R²), also called the coefficient of determination, is:

- A measure of how well your model explains the variation in your data
- Expressed as a value typically between 0 and 1 (or 0% to 100%)
- A higher value means your model explains more of the variation in your data

How R-Squared works:

- R² = 1 means your model perfectly explains all the variation in your data
- R² = 0 means your model doesn't explain any of the variation (it's no better than just predicting the average)
- R² can sometimes be negative if your model performs worse than just predicting the average

Think of it like this:
Imagine you're predicting house prices:

- If R² = 0.7, it means 70% of the variation in house prices is explained by your model
- The remaining 30% is due to factors your model doesn't capture

A simple way to understand R²:

- If you always predicted the average house price, you'd have an R² of 0
- If you could predict every house price exactly right, you'd have an R² of 1
- Your model's R² tells you how much better it is than just predicting the average

Key characteristics of R²:

- Higher values are better (1 would be perfect)
- It's scale-independent (it doesn't matter if you're predicting dollars or millions of dollars)
- It helps you understand how much of the variation your model captures
- It can be misleading if you have a small sample size or too many features

When to use R²:

- When you want to know how much of the variation your model explains
- When you want a metric that's easy to interpret (0% to 100% explained)
- When comparing different models for the same problem
- For regression problems (predicting continuous values)

R² is one of the most popular metrics for regression tasks because it provides an
intuitive measure of how well your model captures the patterns in your data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RSquaredFitnessCalculator(DataSetType)` | Initializes a new instance of the RSquaredFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the R-Squared (R²) fitness score for the given dataset. |

