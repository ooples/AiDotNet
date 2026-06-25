---
title: "IDataSplitter<T>"
description: "Interface for data splitting strategies that divide datasets into train/validation/test sets."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Preprocessing.DataPreparation`

Interface for data splitting strategies that divide datasets into train/validation/test sets.

## For Beginners

Think of it like studying for an exam:

- Training set = Your study materials
- Validation set = Practice tests you use to check understanding
- Test set = The actual exam that determines your grade

## How It Works

**What is Data Splitting?**
Data splitting divides your dataset into separate subsets for different purposes:

- **Training set:** Data the model learns from (typically 60-80%)
- **Validation set:** Data used to tune hyperparameters (typically 10-20%)
- **Test set:** Data for final evaluation (typically 10-20%)

**Why Split Data?**
If you train and test on the same data, you can't tell if your model actually learned
generalizable patterns or just memorized the training examples. Splitting ensures
you evaluate on data the model has never seen.

**Data Splitting vs Data Preprocessing**

- **Data Splitting** (this interface): Changes the NUMBER of rows by dividing data into subsets
- **Data Preprocessing**: Transforms values (scaling, encoding) WITHOUT changing row count

Data splitting is part of Data Preparation (along with outlier removal and augmentation)
and only happens during training, never during prediction.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets a human-readable description of the splitting strategy. |
| `NumSplits` | Gets the number of splits this splitter generates. |
| `RequiresLabels` | Gets whether this splitter requires target labels (y) to function. |
| `SupportsValidation` | Gets whether this splitter supports providing a validation set. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` | Generates multiple train/test splits for cross-validation methods. |
| `GetTensorSplits(Tensor<>,Tensor<>)` | Generates multiple train/test splits for cross-validation on Tensor data. |
| `Split(Matrix<>,Vector<>)` | Performs a single train/test (and optionally validation) split on Matrix data. |
| `SplitTensor(Tensor<>,Tensor<>)` | Performs a single train/test (and optionally validation) split on Tensor data. |

