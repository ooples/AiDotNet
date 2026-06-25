---
title: "SimpleRegression"
description: "Implements simple linear regression, which predicts a single output value based on a single input feature."
section: "Reference"
---

_Regression Models_

Implements simple linear regression, which predicts a single output value based on a single input feature. This is the most basic form of regression that finds the best-fitting straight line through a set of points.

## For Beginners

Simple linear regression is like drawing the best straight line through a set of points. Think of it like this: - You have data points on a graph (like house sizes and their prices) - You want to find the line that best represents the relationship - This line helps you predict new values (like the price of a house based on its size) For example, if you plot people's heights and weights, simple regression would find the line that shows how weight typically increases with height, allowing you to estimate someone's weight if you only know their height.

## How It Works

Simple linear regression models the relationship between two variables by fitting a linear equation: y = mx + b where: - y is the predicted output value - x is the input feature value - m is the slope (coefficient) - b is the y-intercept (where the line crosses the y-axis)

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
    .ConfigureModel(new SimpleRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained SimpleRegression.");
```

