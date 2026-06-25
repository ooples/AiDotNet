---
title: "TweedieRegression"
description: "Implements Tweedie regression, a flexible generalized linear model that encompasses several distributions (Poisson, Gamma, Inverse Gaussian) as special cases based on the power parameter."
section: "Reference"
---

_Regression Models_

Implements Tweedie regression, a flexible generalized linear model that encompasses several distributions (Poisson, Gamma, Inverse Gaussian) as special cases based on the power parameter.

## How It Works

Tweedie regression is a powerful family of distributions where variance is proportional to a power of the mean: Var(Y) = φ × μ^p. The power parameter p determines which distribution family is used: - p = 0: Normal/Gaussian (variance independent of mean) - p = 1: Poisson (variance = mean) - 1 < p < 2: Compound Poisson-Gamma (handles both zeros and positive continuous values) - p = 2: Gamma (variance = mean²) - p = 3: Inverse Gaussian (variance = mean³) 

The compound Poisson-Gamma case (1 < p < 2) is particularly important for insurance modeling, where data often has many exact zeros (no claim) mixed with positive continuous values (claim amounts). 

The model is fitted using iteratively reweighted least squares (IRLS), a form of maximum likelihood estimation. 

For Beginners: Tweedie regression is like having a "dial" that lets you choose how the variability in your data relates to the average. It's especially powerful because: - Insurance claims: Many policies have zero claims, others have positive amounts - Rainfall data: Many dry days (zero) plus positive rainfall amounts - Healthcare costs: Some patients have zero costs, others have positive costs - Sales data: Some products have zero sales, others have positive sales With p between 1 and 2, Tweedie can naturally handle data that has both exact zeros and positive continuous values - something that neither Poisson (counts only) nor Gamma (positive only) can do alone.

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
    .ConfigureModel(new TweedieRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained TweedieRegression.");
```

