---
title: "StratifiedKFoldCrossValidator<T, TInput, TOutput, TMetadata>"
description: "Implements a stratified k-fold cross-validation strategy for model evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CrossValidators`

Implements a stratified k-fold cross-validation strategy for model evaluation.

## For Beginners

Stratified k-fold cross-validation is like k-fold, but it ensures that each fold
has roughly the same proportion of different types of data as the whole dataset.

What this class does:

- Splits your data into k parts (folds), maintaining the balance of different classes in each fold
- Uses each part once for testing and the rest for training
- Repeats this process k times, so each part gets a chance to be the test set
- Calculates how well your model performs on average across all these tests

This is particularly useful when:

- Your data has imbalanced classes (some types of data are much more common than others)
- You want to ensure each fold is representative of the overall dataset

## How It Works

This class provides a stratified k-fold cross-validation implementation, where the data is split into k folds
while maintaining the proportion of samples for each class.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedKFoldCrossValidator(CrossValidationOptions)` | Initializes a new instance of the StratifiedKFoldCrossValidator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFolds(,)` | Creates the stratified folds for k-fold cross-validation based on the provided options. |
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs the stratified k-fold cross-validation process on the given model using the provided data and optimizer. |

