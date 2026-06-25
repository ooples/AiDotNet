---
title: "PreprocessingPipeline<T, TInput, TOutput>"
description: "Chains multiple data transformers into a sequential pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing`

Chains multiple data transformers into a sequential pipeline.

## For Beginners

Think of a pipeline as a series of steps.
For example:

1. First, fill in missing values (SimpleImputer)
2. Then, scale the data (StandardScaler)
3. Finally, create polynomial features (PolynomialFeatures)

The pipeline ensures all these steps happen in the right order.

## How It Works

A preprocessing pipeline applies transformers in sequence, passing the output
of each transformer as input to the next. This enables composable preprocessing
workflows similar to sklearn's Pipeline.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PreprocessingPipeline` | Creates a new empty preprocessing pipeline. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` | Gets the column indices this transformer operates on (null for pipelines). |
| `Count` | Gets the number of steps in the pipeline. |
| `IsFitted` | Gets whether this pipeline has been fitted to data. |
| `Steps` | Gets the named steps in the pipeline. |
| `SupportsInverseTransform` | Gets whether this pipeline supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IDataTransformer<,,>)` | Adds a transformer step to the pipeline. |
| `Add(String,IDataTransformer<,,>)` | Adds a named transformer step to the pipeline. |
| `Clone` | Creates a clone of this pipeline without fitted state. |
| `Fit()` | Fits all transformers in the pipeline to the training data. |
| `FitTransform()` | Fits the pipeline and transforms the data in a single step. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after all transformations. |
| `GetStep(String)` | Gets a transformer step by name. |
| `InverseTransform()` | Inverse transforms data through all pipeline steps in reverse order. |
| `SetFinalTransformer(IDataTransformer<,,>)` | Sets the final transformer that may change the output type. |
| `Transform()` | Transforms data through all pipeline steps. |

