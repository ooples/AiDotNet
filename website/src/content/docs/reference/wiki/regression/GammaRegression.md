---
title: "GammaRegression"
description: "Implements Gamma regression, a generalized linear model for positive continuous data with right-skewed distributions."
section: "Reference"
---

_Regression Models_

Implements Gamma regression, a generalized linear model for positive continuous data with right-skewed distributions.

## How It Works

Gamma regression is appropriate when the response variable is positive continuous and often right-skewed,
with variance that increases with the mean. It's commonly used for modeling durations, costs, and other
positive quantities where the coefficient of variation is approximately constant.

The Gamma distribution has two parameters: shape (k) and scale (θ), with:

- Mean: μ = k × θ
- Variance: μ²/k = φ × μ², where φ = 1/k is the dispersion parameter

The model is fitted using iteratively reweighted least squares (IRLS), a form of maximum likelihood estimation.

For Beginners:
Gamma regression is used when you're trying to predict positive continuous values that are often right-skewed
(meaning most values are small but some are very large). Common examples include:

- Insurance claim amounts
- Hospital length of stay
- Income levels
- Time until an event occurs
- Costs and prices

Unlike linear regression which can predict negative values, Gamma regression ensures predictions are always
positive. It also handles the common pattern where larger values tend to be more variable.

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
    .ConfigureModel(new GammaRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained GammaRegression.");
```

