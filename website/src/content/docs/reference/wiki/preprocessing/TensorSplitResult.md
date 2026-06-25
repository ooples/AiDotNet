---
title: "TensorSplitResult<T>"
description: "Contains the results of a data split operation for Tensor data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation`

Contains the results of a data split operation for Tensor data.

## For Beginners

This is the same as DataSplitResult, but for tensor (multi-dimensional) data
like images, sequences, or other complex structures.

## Properties

| Property | Summary |
|:-----|:--------|
| `FoldIndex` | Gets the fold index for cross-validation methods (0-based). |
| `HasValidationSet` | Gets whether this result includes a validation set. |
| `RepeatIndex` | Gets the repeat index for repeated cross-validation methods (0-based). |
| `TestIndices` | Gets the indices of samples from the original data that are in the test set. |
| `TotalFolds` | Gets the total number of folds for cross-validation methods. |
| `TotalRepeats` | Gets the total number of repeats for repeated cross-validation methods. |
| `TrainIndices` | Gets the indices of samples from the original data that are in the training set. |
| `ValidationIndices` | Gets the indices of samples from the original data that are in the validation set (optional). |
| `XTest` | Gets the feature tensor for testing. |
| `XTrain` | Gets the feature tensor for training. |
| `XValidation` | Gets the feature tensor for validation (optional three-way split). |
| `yTest` | Gets the target tensor for testing (null for unsupervised learning). |
| `yTrain` | Gets the target tensor for training (null for unsupervised learning). |
| `yValidation` | Gets the target tensor for validation (optional three-way split). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Gets a human-readable summary of this split result. |

