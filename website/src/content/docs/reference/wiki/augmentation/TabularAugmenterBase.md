---
title: "TabularAugmenterBase<T>"
description: "Base class for tabular data augmentations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Augmentation.Tabular`

Base class for tabular data augmentations.

## For Beginners

Tabular augmentation transforms structured data (like spreadsheets)
to improve model generalization. Unlike image augmentation which uses geometric transforms,
tabular augmentation focuses on:

- Adding noise to numerical features
- Mixing samples together (MixUp)
- Dropping features for regularization
- Synthetic sample generation (SMOTE)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabularAugmenterBase(Double)` | Initializes a new tabular augmentation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFeatureCount(Matrix<>)` | Gets the number of features in the input data. |
| `GetSampleCount(Matrix<>)` | Gets the number of samples in the input data. |

