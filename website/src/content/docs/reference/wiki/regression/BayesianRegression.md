---
title: "BayesianRegression"
description: "Implements Bayesian Linear Regression with support for various kernels and uncertainty estimation."
section: "Reference"
---

_Regression Models_

Implements Bayesian Linear Regression with support for various kernels and uncertainty estimation.

## For Beginners

Bayesian regression is a special type of regression model that not only predicts values
but also tells you how confident it is about those predictions.

Think of it this way: If you were to guess someone's weight just by looking at their height, you wouldn't
be 100% sure about your guess. You'd have some uncertainty. Bayesian regression captures this uncertainty
mathematically.

Key features of Bayesian regression:

- It calculates probabilities instead of just point estimates
- It can tell you which predictions are more reliable than others
- It combines prior knowledge with observed data to make inferences
- It can incorporate various "kernels" to model different types of relationships

A "kernel" is like a special lens that transforms how the model sees relationships in your data.
For example, some kernels are good at capturing curved relationships, while others might be better
for periodic patterns.

Bayesian regression is especially useful when:

- You have limited data
- You want to know how confident the model is in its predictions
- You need to incorporate prior knowledge about the problem

## How It Works

Bayesian Linear Regression extends traditional linear regression by using Bayesian inference to provide
a probabilistic model of the regression problem. Instead of point estimates of the model parameters,
it computes a full posterior distribution over the parameters, allowing for uncertainty quantification
in predictions. The model assumes Gaussian prior distributions on the parameters and Gaussian noise
in the observations.

This implementation supports various kernel functions for non-linear regression, including:

- Linear kernel (standard linear regression)
- Radial Basis Function (RBF) kernel
- Polynomial kernel
- Sigmoid kernel
- Laplacian kernel

The choice of kernel enables the model to capture different types of relationships between features and targets.

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
    .ConfigureModel(new BayesianRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained BayesianRegression.");
```

