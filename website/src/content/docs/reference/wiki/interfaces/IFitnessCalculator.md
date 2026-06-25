---
title: "IFitnessCalculator<T, TInput, TOutput>"
description: "Defines an interface for calculating how well a machine learning model performs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for calculating how well a machine learning model performs.

## How It Works

**For Beginners:** This interface helps measure how "fit" or effective your machine learning model is.

In machine learning, we need a way to measure how good our model is at making predictions.
This is similar to how we might grade a test - we need a scoring system.

Different types of problems need different scoring methods:

- For predicting house prices, we might measure how close our predictions are to actual prices
- For classifying emails as spam/not spam, we might count how many emails we classified correctly

The "fitness score" is this measurement of how well the model performs. Some important points:

- Sometimes higher scores are better (like accuracy: 95% is better than 90%)
- Sometimes lower scores are better (like error: 5% error is better than 10% error)
- The score helps us compare different models to choose the best one

This interface provides methods to calculate these fitness scores in a standardized way.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsHigherScoreBetter` | Indicates whether higher fitness scores represent better performance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFitnessScore(DataSetStats<,,>)` | Calculates a fitness score based on basic dataset statistics. |
| `CalculateFitnessScore(ModelEvaluationData<,,>)` | Calculates a fitness score based on comprehensive model evaluation data. |
| `IsBetterFitness(,)` | Compares two fitness scores and determines if the current score is better than the best score so far. |

