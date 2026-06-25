---
title: "SplineRegression"
description: "Implements spline regression, which models nonlinear relationships by fitting piecewise polynomial functions."
section: "Reference"
---

_Regression Models_

Implements spline regression, which models nonlinear relationships by fitting piecewise polynomial functions. This advanced regression technique offers more flexibility than simple linear regression by allowing the model to change its behavior across different regions of the data.

## For Beginners

Spline regression is like drawing a curve through your data that can bend and adjust at specific points (called knots). Think of it like this: - Instead of forcing a single straight line through all your data - The model places connection points (knots) where the curve can change direction - These knots let the model adapt to different patterns in different regions of your data For example, if modeling how temperature affects plant growth: - Below freezing: plants don't grow at all (flat line) - From freezing to optimal: growth increases rapidly (steep curve) - Above optimal: growth slows again (flatter curve) A spline regression can capture these changing relationships much better than a simple line.

## How It Works

Spline regression uses basis functions centered at specific points called knots. The model combines: - A constant term - Polynomial terms of the input features (up to a specified degree) - Spline terms that activate beyond each knot This creates a piecewise function that can smoothly adapt to local patterns in the data.

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

