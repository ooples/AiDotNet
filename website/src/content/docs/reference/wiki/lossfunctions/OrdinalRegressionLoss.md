---
title: "OrdinalRegressionLoss<T>"
description: "Implements the Ordinal Regression Loss function for predicting ordered categories."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Ordinal Regression Loss function for predicting ordered categories.

## For Beginners

Ordinal Regression is used when predicting categories that have a meaningful order.
Examples include:

- Ratings (poor, fair, good, excellent)
- Education levels (elementary, middle, high school, college)
- Severity levels (mild, moderate, severe)

Unlike regular classification, ordinal regression takes into account that being off by one category
is better than being off by multiple categories. For example, predicting "good" when the actual 
rating is "fair" is a smaller error than predicting "excellent".

The ordinal regression loss uses a series of binary classifiers, one for each threshold between
adjacent categories. For example, with categories [1,2,3,4,5], there are four classifiers:

- Is the rating > 1?
- Is the rating > 2?
- Is the rating > 3?
- Is the rating > 4?

This approach preserves the ordering information in the categories during training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrdinalRegressionLoss(Int32)` | Initializes a new instance of the OrdinalRegressionLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Ordinal Regression Loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Ordinal Regression Loss between predicted and actual values. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | Small value to prevent numerical instability. |
| `_numClasses` | The number of classes or categories in the ordinal scale. |

