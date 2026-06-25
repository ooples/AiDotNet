---
title: "SplitCriterion"
description: "Specifies the criterion used to determine the best way to split data in decision trees and other tree-based models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the criterion used to determine the best way to split data in decision trees and other tree-based models.

## How It Works

**For Beginners:** Split criteria are like the "rules" that help decision trees decide how to divide data.

Imagine you're organizing books on shelves. You could sort them by:

- Size (big books vs. small books)
- Color (red books vs. blue books)
- Topic (fiction vs. non-fiction)

But which way of sorting is best? Split criteria help the AI decide which way of dividing
the data will lead to the most accurate predictions. Different criteria measure "best" in
different ways, each with their own advantages.

These criteria are primarily used for regression problems (predicting numeric values like
house prices or temperatures) rather than classification problems (predicting categories).

## Fields

| Field | Summary |
|:-----|:--------|
| `FriedmanMSE` | Selects splits using Friedman's improvement to MSE, which accounts for the potential improvement from further splits. |
| `MeanAbsoluteError` | Selects splits that minimize the mean absolute error between the actual and predicted values. |
| `MeanSquaredError` | Selects splits that minimize the mean squared error between the actual and predicted values. |
| `VarianceReduction` | Selects splits that maximize the reduction in variance of the target variable. |

