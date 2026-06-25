---
title: "NestedCrossValidator<T, TInput, TOutput>"
description: "Implements a nested cross-validation strategy for model evaluation and hyperparameter tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CrossValidators`

Implements a nested cross-validation strategy for model evaluation and hyperparameter tuning.

## For Beginners

Nested cross-validation is like a two-layer testing process for your model.

What this class does:

- Splits your data into outer folds for overall model assessment
- For each outer fold:
- Further splits the training data into inner folds for hyperparameter tuning
- Uses the inner folds to find the best hyperparameters
- Trains a model with the best hyperparameters on the full outer training set
- Evaluates this model on the outer test set
- Calculates how well your model performs on average across all outer folds

This is useful because:

- It helps you choose the best hyperparameters for your model
- It provides an unbiased estimate of your model's performance on new data
- It helps prevent overfitting during the model selection process

## How It Works

This class provides a nested cross-validation implementation, which consists of an outer loop for model assessment
and an inner loop for model selection (hyperparameter tuning).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NestedCrossValidator(ICrossValidator<,,>,ICrossValidator<,,>,Func<CrossValidationResult<,,>,IFullModel<,,>>,CrossValidationOptions)` | Initializes a new instance of the NestedCrossValidator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs the nested cross-validation process on the given model using the provided data and optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_innerValidator` | The cross-validator used for the inner loop of nested cross-validation. |
| `_modelSelector` | A function that selects the best model based on inner cross-validation results. |
| `_outerValidator` | The cross-validator used for the outer loop of nested cross-validation. |

