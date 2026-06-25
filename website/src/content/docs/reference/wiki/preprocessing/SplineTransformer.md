---
title: "SplineTransformer<T>"
description: "Generates B-spline basis functions for features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureGeneration`

Generates B-spline basis functions for features.

## For Beginners

B-splines let your model capture curved (non-linear) patterns:

- Linear models only find straight-line relationships
- Splines create multiple smooth curves that join at "knots"
- Each input feature becomes multiple features representing different curve segments

Example: Age effect on income might be curved (rises until 50, then plateaus).
Splines capture this without needing polynomial features.

## How It Works

SplineTransformer creates B-spline basis functions from input features. B-splines
are piecewise polynomials that provide a flexible way to model non-linear relationships
while maintaining smoothness.

The knots can be placed uniformly across the feature range or at quantile positions
to ensure roughly equal numbers of samples between knots.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SplineTransformer(Int32,Int32,SplineKnotStrategy,Boolean,SplineExtrapolation,Int32[])` | Creates a new instance of `SplineTransformer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Degree` | Gets the degree of the spline. |
| `Extrapolation` | Gets the extrapolation strategy. |
| `IncludeIntercept` | Gets whether the intercept term is included. |
| `KnotStrategy` | Gets the knot placement strategy. |
| `Knots` | Gets the fitted knots for each feature. |
| `NKnots` | Gets the number of internal knots. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the knot positions for each feature. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for spline basis functions. |
| `TransformCore(Matrix<>)` | Transforms the data by computing B-spline basis values. |

