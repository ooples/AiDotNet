---
title: "MonteCarloValidator<T, TInput, TOutput>"
description: "Implements a Monte Carlo cross-validation strategy for model evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CrossValidators`

Implements a Monte Carlo cross-validation strategy for model evaluation.

## For Beginners

Monte Carlo cross-validation is like repeatedly shuffling and splitting your data to test your model.

What this class does:

- Randomly splits your data into training and validation sets
- Repeats this process multiple times (as specified in the options)
- For each split:
- Trains the model on the training set
- Evaluates the model on the validation set
- Calculates how well your model performs on average across all splits

This is useful because:

- It provides a robust estimate of model performance
- It helps to reduce the impact of how the data is split on the results
- It can be more flexible than k-fold cross-validation for certain types of data

However, it can be computationally expensive for a large number of iterations.

## How It Works

This class provides a Monte Carlo cross-validation implementation, which randomly splits the data into training and validation sets multiple times.

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Initializes a new instance of the MonteCarloValidator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFolds(,)` | Creates the folds for Monte Carlo cross-validation. |
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs the Monte Carlo cross-validation process on the given model using the provided data and optimizer. |

