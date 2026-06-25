---
title: "FeatureUnion<T>"
description: "Concatenates results from multiple transformers horizontally."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing`

Concatenates results from multiple transformers horizontally.

## For Beginners

Sometimes you want multiple feature sets from the same data:

- Polynomial features from numeric columns
- Statistics (mean, std) from time windows
- Both PCA and manual feature engineering

FeatureUnion runs all transformers and combines their outputs side by side.

## How It Works

FeatureUnion applies multiple transformers to the same input data and
concatenates their outputs into a single feature matrix. This is useful
for combining different feature extraction methods.

Each transformer receives the full input matrix and produces its own
output. All outputs are then concatenated column-wise.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureUnion` | Creates a new instance of `FeatureUnion`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IDataTransformer<,Matrix<>,Matrix<>>)` | Adds a transformer to the union. |
| `Add(String,IDataTransformer<,Matrix<>,Matrix<>>)` | Adds a transformer to the union. |
| `FitCore(Matrix<>)` | Fits all transformers to the input data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetTransformer(String)` | Gets the transformer with the specified name. |
| `GetTransformerNames` | Gets all transformer names. |
| `GetTransformerOutputWidths` | Gets the number of output features from each transformer. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for FeatureUnion. |
| `TransformCore(Matrix<>)` | Transforms the data by applying all transformers and concatenating outputs. |

