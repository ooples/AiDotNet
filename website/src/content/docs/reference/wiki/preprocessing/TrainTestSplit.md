---
title: "TrainTestSplit<T>"
description: "Provides simple, standalone utility methods for splitting data into training, validation, and test sets."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Preprocessing`

Provides simple, standalone utility methods for splitting data into training, validation, and test sets.

## For Beginners

Before training a machine learning model, you need to split your data:

- Training set: The data the model learns from (typically 70-80%)
- Test set: The data used to evaluate how well the model performs on unseen data (typically 20-30%)
- Validation set (optional): Data used to tune hyperparameters during training (typically 10-15%)

This class provides simple static methods to perform these splits without needing to configure
a full data preprocessor.

## Methods

| Method | Summary |
|:-----|:--------|
| `KFoldSplit(Matrix<>,Vector<>,Int32,Boolean,Int32)` | Splits data into k folds for cross-validation. |
| `Split(Matrix<>,Vector<>,Double,Boolean,Int32)` | Splits data into training and test sets. |
| `SplitThreeWay(Matrix<>,Vector<>,Double,Double,Boolean,Int32)` | Splits data into training, validation, and test sets. |
| `SplitX(Matrix<>,Double,Boolean,Int32)` | Splits feature matrix only (without targets) into training and test sets. |

