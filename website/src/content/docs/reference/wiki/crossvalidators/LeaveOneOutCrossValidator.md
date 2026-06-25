---
title: "LeaveOneOutCrossValidator<T, TInput, TOutput>"
description: "Implements a Leave-One-Out cross-validation strategy for model evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CrossValidators`

Implements a Leave-One-Out cross-validation strategy for model evaluation.

## For Beginners

Leave-One-Out cross-validation is like testing your model on each data point individually.

What this class does:

- For each data point in your dataset:
- Uses that single point for testing
- Uses all other points for training
- Repeats this process for every single data point
- Calculates how well your model performs on average across all these tests

This is useful because:

- It uses almost all of your data for training in each iteration
- It gives you a performance estimate for each individual data point
- It's particularly useful for small datasets

However, it can be computationally expensive for large datasets.

## How It Works

This class provides a Leave-One-Out cross-validation implementation, where each data point is used once as the validation set
while the remaining data points form the training set.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeaveOneOutCrossValidator(CrossValidationOptions)` | Initializes a new instance of the LeaveOneOutCrossValidator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFolds(,)` | Creates the folds for leave-one-out cross-validation. |
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs the leave-one-out cross-validation process on the given model using the provided data and optimizer. |

