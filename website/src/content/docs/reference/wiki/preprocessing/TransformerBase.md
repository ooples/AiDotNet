---
title: "TransformerBase<T, TInput, TOutput>"
description: "Abstract base class for all data transformers providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Preprocessing`

Abstract base class for all data transformers providing common functionality.

## For Beginners

This is the foundation that all transformers build on.
It provides common features like:

- Checking if the transformer is ready to use
- Managing which columns to transform
- Serialization for saving/loading transformers

When creating a new transformer, you extend this class and implement the abstract methods.

## How It Works

This class provides the template method pattern for data transformation.
Derived classes implement the core fitting and transformation logic while
this base class handles validation, state management, and common operations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerBase(Int32[])` | Creates a new instance of the transformer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` | Gets the column indices this transformer operates on. |
| `Engine` | Gets the computational engine for tensor operations. |
| `IsFitted` | Gets whether this transformer has been fitted to data. |
| `NumOps` | Gets the numeric operations helper for type T. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsureFitted` | Ensures the transformer has been fitted before transformation. |
| `Fit()` | Fits the transformer to the training data. |
| `FitCore()` | Core fitting implementation. |
| `FitTransform()` | Fits the transformer and transforms the data in a single step. |
| `GetColumnsToProcess(Int32)` | Gets the indices to operate on, defaulting to all if ColumnIndices is null. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransform()` | Reverses the transformation. |
| `InverseTransformCore()` | Core inverse transformation implementation. |
| `Transform()` | Transforms the input data using fitted parameters. |
| `TransformCore()` | Core transformation implementation. |
| `ValidateInputData()` | Validates input data before fitting or transforming. |

