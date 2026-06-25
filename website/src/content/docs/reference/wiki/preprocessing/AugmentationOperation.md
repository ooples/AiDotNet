---
title: "AugmentationOperation<T>"
description: "A row operation that applies data augmentation to increase dataset size."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation`

A row operation that applies data augmentation to increase dataset size.

## For Beginners

Data augmentation creates new synthetic data points based on
your existing data. This is especially useful when:

- You have imbalanced classes (one class has way more samples than another)
- You have limited training data
- You want to reduce overfitting

## How It Works

This operation wraps tabular augmenters (like SMOTE) to generate synthetic samples.
Both features (X) and labels (y) are augmented together to maintain alignment.

**Common Use Case - SMOTE:** If you're predicting fraud (rare) vs normal (common),
SMOTE creates synthetic fraud examples so your model learns to recognize fraud better.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AugmentationOperation(TabularAugmenterBase<>,Vector<>)` | Creates a new augmentation operation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Augmenter` | Gets the underlying augmenter. |
| `Description` |  |
| `IsFitted` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitResample(Matrix<>,Vector<>)` |  |
| `FitResampleTensor(Tensor<>,Tensor<>)` |  |

