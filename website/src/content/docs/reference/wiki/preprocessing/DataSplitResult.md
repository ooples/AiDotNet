---
title: "DataSplitResult<T>"
description: "Contains the results of a data split operation for Matrix/Vector data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation`

Contains the results of a data split operation for Matrix/Vector data.

## For Beginners

After splitting your data, this class holds all the pieces:

- Training data (XTrain, yTrain): What your model learns from
- Test data (XTest, yTest): What you evaluate your model on
- Validation data (XValidation, yValidation): Optional, for hyperparameter tuning
- Indices: Which rows from the original data ended up in each set

## How It Works

**Why Track Indices?**
Knowing which original samples are in each set is useful for:

- Debugging: "Why did my model get sample #42 wrong?"
- Analysis: Comparing predictions to original data
- Reproducibility: Recording exactly how data was split

## Properties

| Property | Summary |
|:-----|:--------|
| `FoldIndex` | Gets the fold index for cross-validation methods (0-based). |
| `HasValidationSet` | Gets whether this result includes a validation set. |
| `RepeatIndex` | Gets the repeat index for repeated cross-validation methods (0-based). |
| `TestIndices` | Gets the indices of rows from the original data that are in the test set. |
| `TotalFolds` | Gets the total number of folds for cross-validation methods. |
| `TotalRepeats` | Gets the total number of repeats for repeated cross-validation methods. |
| `TrainIndices` | Gets the indices of rows from the original data that are in the training set. |
| `ValidationIndices` | Gets the indices of rows from the original data that are in the validation set (optional). |
| `XTest` | Gets the feature matrix for testing. |
| `XTrain` | Gets the feature matrix for training. |
| `XValidation` | Gets the feature matrix for validation (optional three-way split). |
| `yTest` | Gets the target vector for testing (null for unsupervised learning). |
| `yTrain` | Gets the target vector for training (null for unsupervised learning). |
| `yValidation` | Gets the target vector for validation (optional three-way split). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Gets a human-readable summary of this split result. |

