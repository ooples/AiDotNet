---
title: "DataSplitter"
description: "Simple utility for splitting data into train/validation/test sets."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Preprocessing.DataPreparation`

Simple utility for splitting data into train/validation/test sets.

## For Beginners

Before training a model, you need to split your data:

- **Training set:** The data your model learns from (typically 70-80%)
- **Validation set:** Used to tune hyperparameters and prevent overfitting (typically 10-15%)
- **Test set:** Final evaluation of model performance (typically 10-15%)

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSplitSizes(Int32,Double,Double)` | Computes train/validation/test partition sizes that always sum to `totalSamples` and never leave a requested partition empty when the data can afford it. |
| `Split(,,Double,Double,Boolean,Int32)` | Splits data into training, validation, and test sets. |

