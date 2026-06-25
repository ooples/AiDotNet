---
title: "TimeSeriesCrossValidator<T, TInput, TOutput>"
description: "Implements a time series cross-validation strategy for model evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CrossValidators`

Implements a time series cross-validation strategy for model evaluation.

## For Beginners

Time series cross-validation is designed for data that has a time component.

What this class does:

- Starts with a small portion of your data for training
- Uses the next part for validation
- Expands the training set to include the previous validation set
- Repeats this process, moving forward in time
- Calculates how well your model performs on average across all these tests

This is useful because:

- It respects the time order of your data
- It simulates how the model would perform in a real-world scenario where you use past data to predict the future
- It helps detect if your model's performance changes over time

## How It Works

This class provides a time series cross-validation implementation, which respects the temporal order of the data.
It uses an expanding window approach, where the training set grows over time.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesCrossValidator(Int32,Int32,Int32,CrossValidationOptions)` | Initializes a new instance of the TimeSeriesCrossValidator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFolds(,)` | Creates the folds for time series cross-validation based on the provided parameters. |
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs the time series cross-validation process on the given model using the provided data and optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_initialTrainSize` | The initial size of the training set. |
| `_step` | The step size for expanding the training set. |
| `_validationSize` | The size of the validation set. |

