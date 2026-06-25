---
title: "OrdinalRegressionLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Ordinal Regression Loss to evaluate model performance, particularly for ordinal classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Ordinal Regression Loss to evaluate model performance, particularly for ordinal classification tasks.

## For Beginners

This calculator helps evaluate how well your model is performing on ordinal classification tasks,
which are a special type of classification where the categories have a meaningful order or rank.

Ordinal Regression Loss is designed specifically for problems where:

- You're predicting categories that have a natural order
- The distance between categories matters
- You want to penalize predictions that are further from the true category more heavily

Examples of ordinal classification problems:

- Rating predictions (1-5 stars)
- Education levels (elementary, middle, high school, college)
- Customer satisfaction levels (very dissatisfied, dissatisfied, neutral, satisfied, very satisfied)
- Disease severity (mild, moderate, severe)

How Ordinal Regression Loss works:

- It recognizes that predicting a 4 when the true value is 5 is better than predicting a 1
- It penalizes predictions based on how far they are from the true category
- It takes into account the ordered nature of your categories

Think of it like this:
Imagine you're predicting movie ratings (1-5 stars):

- Predicting 4 stars when the actual rating is 5 stars is a small error
- Predicting 1 star when the actual rating is 5 stars is a large error
- Ordinal Regression Loss will penalize the second error much more heavily

Key characteristics:

- It's specifically designed for ordered categories
- It penalizes errors based on the distance between predicted and actual categories
- Lower values are better (0 would be perfect predictions)
- It can automatically detect if your problem is suitable for ordinal regression

This calculator is smart enough to:

- Use ordinal regression loss when appropriate
- Fall back to other loss functions when your data doesn't fit the ordinal pattern

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrdinalRegressionLossFitnessCalculator(Nullable<Int32>,DataSetType)` | Initializes a new instance of the OrdinalRegressionLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DefaultLossCalculation(DataSetStats<,,>)` | Calculates the appropriate loss when the number of classes is not explicitly provided. |
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Ordinal Regression Loss fitness score for the given dataset. |
| `IsClassificationProblem(DataSetStats<,,>)` | Determines whether the dataset represents a classification problem based on its characteristics. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numClasses` | The number of classes or categories in the ordinal classification problem. |

