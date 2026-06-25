---
title: "ITransform<TInput, TOutput>"
description: "Core interface for composable data transforms in the data pipeline."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Data.Transforms`

Core interface for composable data transforms in the data pipeline.

## For Beginners

A transform converts data from one form to another.
For example, normalizing pixel values from [0, 255] to [0, 1], or converting
a label integer to a one-hot encoded vector.

## How It Works

Transforms are deterministic, composable operations applied during data loading.
They differ from augmentations (in `AiDotNet.Augmentation`), which are
stochastic and applied during training.

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply()` | Applies the transform to the input data. |

