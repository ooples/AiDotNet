---
title: "FunctionTransformer<T>"
description: "Applies a custom function to transform data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureGeneration`

Applies a custom function to transform data.

## For Beginners

Sometimes you need a custom transformation that doesn't
fit standard transformers. FunctionTransformer lets you plug in your own function:

- Apply a mathematical formula to all values
- Perform domain-specific feature engineering
- Wrap existing transformation code

Example: Apply a custom normalization formula or domain-specific scaling.

## How It Works

FunctionTransformer allows you to apply arbitrary functions to your data as part
of a preprocessing pipeline. This is useful for applying domain-specific transformations
or wrapping legacy code into the transformer API.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FunctionTransformer(Func<Matrix<>,Matrix<>>,Func<Matrix<>,Matrix<>>,Boolean,Int32[])` | Creates a new instance of `FunctionTransformer` with matrix-level functions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Validate` | Gets whether this transformer validates input/output shapes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Abs` | Creates a FunctionTransformer that applies the absolute value. |
| `Clip(Double,Double)` | Creates a FunctionTransformer that clips values to a range. |
| `Exp` | Creates a FunctionTransformer that applies the exponential function. |
| `FitCore(Matrix<>)` | Stores the input dimensions for validation. |
| `FromElementFunction(Func<Double,Double>,Func<Double,Double>,Boolean,Int32[])` | Creates a new instance of `FunctionTransformer` with element-wise functions. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Applies the inverse transformation function to the data. |
| `Log` | Creates a FunctionTransformer that applies the natural logarithm. |
| `Log1p` | Creates a FunctionTransformer that applies log(1 + x). |
| `Power(Double)` | Creates a FunctionTransformer that raises to a power. |
| `Sign` | Creates a FunctionTransformer that applies the sign function. |
| `Sqrt` | Creates a FunctionTransformer that applies the square root. |
| `TransformCore(Matrix<>)` | Applies the transformation function to the data. |

