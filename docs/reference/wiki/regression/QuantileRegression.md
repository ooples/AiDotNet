---
title: "QuantileRegression"
description: "Implements Quantile Regression, a technique that estimates the conditional quantiles of a response variable distribution in the linear model, providing a more complete view of the relationship between variables."
section: "Reference"
---

_Regression Models_

Implements Quantile Regression, a technique that estimates the conditional quantiles of a response variable
distribution in the linear model, providing a more complete view of the relationship between variables.

## For Beginners

While standard regression tells you about the average relationship between variables, quantile regression
lets you explore different parts of the data distribution. For example, median regression (quantile=0.5)
tells you about the middle of the distribution, while quantile=0.9 tells you about the upper end.
This is useful when you suspect that the relationship between variables might be different for different
ranges of the outcome.

## How It Works

Unlike ordinary least squares regression which estimates the conditional mean of the response variable,
quantile regression estimates the conditional median or other quantiles of the response variable.
This makes it robust to outliers and useful for modeling heterogeneous conditional distributions.

The algorithm uses gradient descent optimization to minimize the quantile loss function, which gives
different weights to positive and negative errors based on the specified quantile.

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
    .ConfigureModel(new QuantileRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained QuantileRegression.");
```

