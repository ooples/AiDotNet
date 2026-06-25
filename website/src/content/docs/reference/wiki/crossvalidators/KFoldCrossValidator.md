---
title: "KFoldCrossValidator<T, TInput, TOutput>"
description: "Implements a k-fold cross-validation strategy for model evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CrossValidators`

Implements a k-fold cross-validation strategy for model evaluation.

## For Beginners

K-fold cross-validation is like dividing your data into k equal parts.

What this class does:

- Splits your data into k parts (folds)
- Uses each part once for testing and the rest for training
- Repeats this process k times, so each part gets a chance to be the test set
- Calculates how well your model performs on average across all these tests

This is useful because:

- It uses all of your data for both training and testing
- It gives a more reliable estimate of your model's performance
- It helps detect if your model is overfitting to a particular subset of the data

## How It Works

This class provides a k-fold cross-validation implementation, where the data is split into k equal-sized folds.
Each fold is used once as a validation set while the k-1 remaining folds form the training set.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KFoldCrossValidator(CrossValidationOptions)` | Initializes a new instance of the KFoldCrossValidator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFolds(,)` | Creates the folds for k-fold cross-validation based on the provided options. |
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs the k-fold cross-validation process on the given model using the provided data and optimizer. |

