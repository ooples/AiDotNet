---
title: "IRowOperation<T>"
description: "Defines an operation that can change the number of rows in a dataset."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Preprocessing.DataPreparation`

Defines an operation that can change the number of rows in a dataset.

## For Beginners

Some preprocessing steps need to add or remove entire data points
(rows) from your dataset. When you remove an outlier or create a synthetic sample,
you need to update both the input features AND the corresponding labels together,
otherwise they would become misaligned.

## How It Works

Row operations are fundamentally different from standard transformations because they
modify both features (X) and labels (y) together to maintain alignment. Examples include:

- Outlier removal (reduces rows)
- SMOTE oversampling (adds rows)
- Undersampling (reduces rows)

**Critical:** Row operations are only applied during training (Fit), never during
prediction. This follows the industry standard established by imbalanced-learn.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets a description of what this operation does. |
| `IsFitted` | Gets whether this operation has been fitted to data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitResample(Matrix<>,Vector<>)` | Fits the operation to data and applies the row modification for Matrix/Vector data. |
| `FitResampleTensor(Tensor<>,Tensor<>)` | Fits the operation to data and applies the row modification for Tensor data. |

