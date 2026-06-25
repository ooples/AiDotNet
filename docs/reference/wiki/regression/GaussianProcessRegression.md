---
title: "GaussianProcessRegression"
description: "Implements a Gaussian Process Regression model, which is a non-parametric, probabilistic approach to regression that provides uncertainty estimates along with predictions."
section: "Reference"
---

_Regression Models_

Implements a Gaussian Process Regression model, which is a non-parametric, probabilistic approach to regression that provides uncertainty estimates along with predictions.

## For Beginners

A Gaussian Process Regression model is like a sophisticated way to draw smooth curves through data points. Unlike simpler models that assume a specific shape (like a straight line or parabola), Gaussian Process Regression: - Adapts to fit the data without assuming a predefined shape - Provides not just predictions but also how confident it is in each prediction - Works well with small to medium-sized datasets - Can capture complex patterns in the data You can think of it as drawing a smooth curve through your data points, where the model considers all possible curves that could fit your data and chooses the most likely one based on how similar input points are to each other (defined by a "kernel function"). A unique advantage of this model is that it tells you not just what the prediction is, but also how certain or uncertain that prediction is - like saying "I predict the value is about 42, and I'm pretty confident it's between 40 and 44."

## How It Works

Gaussian Process Regression (GPR) is a flexible, non-parametric approach to regression that models the target function as a sample from a Gaussian process. It provides not only predictions but also uncertainty estimates, making it suitable for applications where quantifying prediction uncertainty is important. The model is defined by a kernel function that determines the covariance between any two points in the input space.

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
    .ConfigureModel(new GaussianProcessRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained GaussianProcessRegression.");
```

