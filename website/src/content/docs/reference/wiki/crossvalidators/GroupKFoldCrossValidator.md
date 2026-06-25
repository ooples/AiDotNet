---
title: "GroupKFoldCrossValidator<T, TInput, TOutput>"
description: "Implements a Group K-Fold cross-validation strategy for model evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CrossValidators`

Implements a Group K-Fold cross-validation strategy for model evaluation.

## For Beginners

Group K-Fold cross-validation is useful when your data has natural groupings.

What this class does:

- Splits your data into k parts (folds) based on group identifiers
- Ensures that all data points from the same group stay together
- Uses each part once for testing and the rest for training
- Repeats this process k times, so each part gets a chance to be the test set
- Calculates how well your model performs on average across all these tests

This is particularly useful when:

- Your data has natural groups (e.g., multiple measurements from the same person)
- You want to ensure that related data points are not split between training and testing sets

## How It Works

This class provides a Group K-Fold cross-validation implementation, where the data is split into k folds
based on a group identifier. This ensures that all samples from the same group are in the same fold.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroupKFoldCrossValidator(Int32[],CrossValidationOptions)` | Initializes a new instance of the GroupKFoldCrossValidator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFolds(,,Int32[])` | Creates the folds for group k-fold cross-validation based on the provided group identifiers. |
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs the group k-fold cross-validation process on the given model using the provided data and optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_groups` | The group identifiers for each sample in the dataset. |

