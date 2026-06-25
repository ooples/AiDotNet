---
title: "IDataTransformer<T, TInput, TOutput>"
description: "Defines a data transformer that can fit to data and transform it."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a data transformer that can fit to data and transform it.

## For Beginners

A transformer is like a recipe that:

1. First "learns" from your training data (e.g., calculates mean and std for scaling)
2. Then applies the same recipe to any new data

This ensures new data is processed exactly the same way as training data.

## How It Works

This is the core interface for all preprocessing transformers in AiDotNet.
It follows the sklearn-style Fit/Transform pattern where transformers first
learn parameters from training data (Fit), then apply transformations (Transform).

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` | Gets the column indices this transformer operates on. |
| `IsFitted` | Gets whether this transformer has been fitted to data. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit()` | Fits the transformer to the training data, learning any parameters needed for transformation. |
| `FitTransform()` | Fits the transformer and transforms the data in a single step. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransform()` | Reverses the transformation (if supported). |
| `Transform()` | Transforms the input data using the fitted parameters. |

