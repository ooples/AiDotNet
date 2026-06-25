---
title: "StandardCrossValidator<T, TInput, TOutput>"
description: "Implements a standard cross-validation strategy for model evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CrossValidators`

Implements a standard cross-validation strategy for model evaluation.

## For Beginners

Cross-validation is like a thorough test for your machine learning model.

What this class does:

- Splits your data into several parts (called folds)
- Trains and tests your model multiple times, each time using a different part as the test set
- Calculates how well your model performs on average across all these tests

This is useful because:

- It helps you understand how well your model will work on new, unseen data
- It can detect if your model is overfitting (memorizing the training data instead of learning general patterns)
- It provides a more reliable estimate of your model's performance than a single train-test split

For example, if you're building a model to predict house prices, cross-validation would test it on different subsets
of your house data, giving you a better idea of how well it will predict prices for houses it hasn't seen before.

## How It Works

This class provides a standard implementation of cross-validation, a technique used to assess how the results of a
statistical analysis will generalize to an independent data set. It is particularly important in contexts where the goal
is prediction, and one wants to estimate how accurately a predictive model will perform in practice.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StandardCrossValidator(CrossValidationOptions)` | Initializes a new instance of the StandardCrossValidator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFolds(,)` | Creates the folds for cross-validation based on the provided options. |
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs the cross-validation process on the given model using the provided data and optimizer. |

