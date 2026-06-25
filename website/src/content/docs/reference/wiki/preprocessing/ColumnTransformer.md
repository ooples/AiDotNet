---
title: "ColumnTransformer<T>"
description: "Applies different transformers to different columns of the input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing`

Applies different transformers to different columns of the input.

## For Beginners

Different columns often need different treatment:

- Numeric columns: scaling, normalization
- Categorical columns: one-hot encoding
- Text columns: vectorization

ColumnTransformer lets you apply the right transformation to each column type.

## How It Works

ColumnTransformer allows you to specify which transformer should be applied
to which columns. This is useful when different columns require different
preprocessing (e.g., scaling numeric columns, encoding categorical columns).

Output columns are concatenated in the order transformers are added.
Columns not specified in any transformer can be passed through or dropped.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ColumnTransformer(ColumnTransformerRemainder)` | Creates a new instance of `ColumnTransformer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Remainder` | Gets how columns not specified in any transformer are handled. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IDataTransformer<,Matrix<>,Matrix<>>,Int32[])` | Adds a transformer to be applied to specific columns. |
| `Add(String,IDataTransformer<,Matrix<>,Matrix<>>,Int32[])` | Adds a transformer to be applied to specific columns. |
| `FitCore(Matrix<>)` | Fits all transformers to their respective columns. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetTransformer(String)` | Gets the transformer with the specified name. |
| `GetTransformerNames` | Gets all transformer names. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for ColumnTransformer. |
| `TransformCore(Matrix<>)` | Transforms the data by applying each transformer to its columns. |

