---
title: "PoissonRegression"
description: "Implements Poisson regression, a generalized linear model used for modeling count data and contingency tables."
section: "Reference"
---

_Regression Models_

Implements Poisson regression, a generalized linear model used for modeling count data and contingency tables.

## How It Works

Poisson regression is appropriate when the dependent variable represents counts, such as the number of occurrences of an event in a fixed period of time or space. It assumes that the response variable follows a Poisson distribution and uses a log link function to relate the expected value of the response to the linear predictor. 

The model is fitted using iteratively reweighted least squares (IRLS), a form of maximum likelihood estimation. 

For Beginners: Poisson regression is used when you're trying to predict counts (like number of customer visits, number of accidents, etc.). Unlike linear regression, it ensures predictions are always non-negative and can handle cases where the variance increases with the mean, which is common in count data.

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
    .ConfigureModel(new PoissonRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained PoissonRegression.");
```

