---
title: "RadialBasisFunctionRegression"
description: "Implements Radial Basis Function (RBF) Regression, a technique that uses radial basis functions as the basis for approximating complex nonlinear relationships between inputs and outputs."
section: "Reference"
---

_Regression Models_

Implements Radial Basis Function (RBF) Regression, a technique that uses radial basis functions
as the basis for approximating complex nonlinear relationships between inputs and outputs.

## For Beginners

Think of RBF regression as placing a set of "bell curves" at strategic locations in your input space.
Each curve gives a strong response when an input is close to its center and a weak response when it's far away.
The model predicts by combining these responses with learned weights. This approach is particularly good at
modeling complex, non-linear relationships in data.

## How It Works

Radial Basis Function Regression works by transforming the input space using a set of radial basis functions,
each centered at a different point. These functions produce a response that depends on the distance from the
input to the center point. The model then combines these responses linearly to make predictions.

The algorithm first selects a set of centers (typically using k-means clustering), computes the RBF features
for each input point, and then solves a linear regression problem to find the optimal weights.

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
    .ConfigureModel(new RadialBasisFunctionRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained RadialBasisFunctionRegression.");
```

