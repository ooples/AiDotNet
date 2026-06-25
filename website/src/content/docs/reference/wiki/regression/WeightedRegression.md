---
title: "WeightedRegression"
description: "Implements weighted regression, a variation of linear regression where each data point has a different  level of importance (weight) in determining the model parameters."
section: "Reference"
---

_Regression Models_

Implements weighted regression, a variation of linear regression where each data point has a different 
level of importance (weight) in determining the model parameters.

## For Beginners

Weighted regression is like giving different voting power to different data points.

Think of it like this:

- Regular regression treats all data points equally - each point gets one "vote" on where the line should go
- Weighted regression lets some points have more "votes" than others
- Points with higher weights have more influence on the final model
- Points with lower weights have less influence

For example, if you're predicting house prices:

- Recent sales might get higher weights because they reflect current market conditions better
- Unusual properties might get lower weights to prevent them from skewing the model
- More reliable measurements might get higher weights than less reliable ones

This helps you build models that focus more on the data points you trust or care about most.

## How It Works

Weighted regression extends standard regression by allowing each data point to have a different level
of influence on the model. This is particularly useful in scenarios where data points have varying
reliability, importance, or error variance.

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
    .ConfigureModel(new WeightedRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained WeightedRegression.");
```

