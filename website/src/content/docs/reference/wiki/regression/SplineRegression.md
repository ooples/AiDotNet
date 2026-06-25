---
title: "SplineRegression<T>"
description: "Implements spline regression, which models nonlinear relationships by fitting piecewise polynomial functions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements spline regression, which models nonlinear relationships by fitting piecewise polynomial functions.
This advanced regression technique offers more flexibility than simple linear regression by allowing the model
to change its behavior across different regions of the data.

## For Beginners

Spline regression is like drawing a curve through your data that can bend and adjust
at specific points (called knots).

Think of it like this:

- Instead of forcing a single straight line through all your data
- The model places connection points (knots) where the curve can change direction
- These knots let the model adapt to different patterns in different regions of your data

For example, if modeling how temperature affects plant growth:

- Below freezing: plants don't grow at all (flat line)
- From freezing to optimal: growth increases rapidly (steep curve)
- Above optimal: growth slows again (flatter curve)

A spline regression can capture these changing relationships much better than a simple line.

## How It Works

Spline regression uses basis functions centered at specific points called knots. The model combines:

- A constant term
- Polynomial terms of the input features (up to a specified degree)
- Spline terms that activate beyond each knot

This creates a piecewise function that can smoothly adapt to local patterns in the data.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new SplineRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained SplineRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SplineRegression(SplineRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Creates a new spline regression model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Spline regression solves analytically — no optimizer parameter injection needed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateInstance` | Creates a new instance of the Spline Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the spline regression model from a byte array. |
| `GenerateBasisFunctions(Matrix<>)` | Generates the basis functions matrix for the input data. |
| `GenerateKnots(Vector<>)` | Generates knots for a single feature vector. |
| `GetActiveFeatureIndices` | Returns original feature indices (not expanded spline basis indices). |
| `OptimizeModel(Matrix<>,Vector<>)` | Optimizes the spline regression model using the provided input data and target values. |
| `Predict(Matrix<>)` | Predicts target values for a matrix of input features. |
| `PredictSingle(Vector<>)` | Predicts a target value for a single input feature vector. |
| `Serialize` | Returns the type identifier for this regression model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The coefficients for the spline model. |
| `_knots` | The collection of knot points for each feature. |
| `_options` | Configuration options for the spline regression model. |

