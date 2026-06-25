---
title: "InverseGaussianRegression"
description: "Implements Inverse Gaussian regression, a generalized linear model for positive continuous data with variance proportional to the cube of the mean."
section: "Reference"
---

_Regression Models_

Implements Inverse Gaussian regression, a generalized linear model for positive continuous data
with variance proportional to the cube of the mean.

## How It Works

The Inverse Gaussian distribution (also known as Wald distribution) is appropriate for modeling
positive continuous response variables with heavy right tails. It's commonly used for modeling
response times, waiting times, and first passage times.

The Inverse Gaussian distribution has two parameters: μ (mean) and λ (shape), with:

- Mean: μ
- Variance: μ³/λ = φ × μ³, where φ = 1/λ is the dispersion parameter

The model is fitted using iteratively reweighted least squares (IRLS), a form of maximum likelihood estimation.

For Beginners:
Inverse Gaussian regression is used when you're trying to predict positive continuous values that have
heavy right tails (meaning extreme large values are possible and have high variability). Common examples include:

- Response times in cognitive experiments
- Time until failure for mechanical systems
- First passage times in physics
- Waiting times in queuing systems

Compared to Gamma regression which has variance proportional to μ², Inverse Gaussian has variance
proportional to μ³, meaning it handles even heavier tails where large values are much more variable.

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
    .ConfigureModel(new InverseGaussianRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained InverseGaussianRegression.");
```

