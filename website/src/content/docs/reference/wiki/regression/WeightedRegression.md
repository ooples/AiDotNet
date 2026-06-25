---
title: "WeightedRegression<T>"
description: "Implements weighted regression, a variation of linear regression where each data point has a different  level of importance (weight) in determining the model parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightedRegression(WeightedRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Creates a new instance of the weighted regression model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the weighted regression model with the same configuration. |
| `ExpandFeatures(Matrix<>)` | Expands the input features to include polynomial terms up to the specified order. |
| `Predict(Matrix<>)` | Makes predictions using the trained weighted regression model. |
| `Train(Matrix<>,Vector<>)` | Trains the weighted regression model using the provided input features and target values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `MinimumStabilityRidge` | Absolute floor for the stability ridge when diagonal magnitude is near zero. |
| `StabilityRidgeScale` | Relative scale factor for the adaptive stability ridge (fraction of mean diagonal magnitude). |
| `_order` | The polynomial order for feature expansion. |
| `_weights` | The weights assigned to each data point, determining their influence on the model. |

