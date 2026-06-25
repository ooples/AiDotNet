---
title: "IsotonicRegression"
description: "Implements an Isotonic Regression model, which fits a free-form line to data with the constraint that the fitted line must be non-decreasing (monotonically increasing)."
section: "Reference"
---

_Regression Models_

Implements an Isotonic Regression model, which fits a free-form line to data with the constraint that the fitted line must be non-decreasing (monotonically increasing).

## For Beginners

Isotonic Regression creates a "stair-step" function that only goes up, never down. Imagine you're drawing a line through points on a graph, but with two key rules: - The line can go up or stay flat, but it can never go down - The line should stick as close as possible to all the data points This model is useful when you know that as one value increases, the other should never decrease. For example: - As study time increases, test scores shouldn't decrease - As price increases, demand shouldn't increase - As age increases, height (for children) shouldn't decrease Unlike a straight line (linear regression), Isotonic Regression can capture more complex relationships while still maintaining this "never decreasing" property.

## How It Works

Isotonic Regression is a form of nonlinear regression that fits a non-decreasing function to data. Unlike many regression techniques, it makes minimal assumptions about the shape of the function besides monotonicity (that the function doesn't decrease as the input increases). This makes it particularly useful for calibrating probability estimates from other models or for situations where a monotonic relationship is expected between the input and output variables.

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
    .ConfigureModel(new IsotonicRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained IsotonicRegression.");
```

