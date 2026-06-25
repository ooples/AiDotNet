---
title: "InverseGaussianRegression<T>"
description: "Implements Inverse Gaussian regression, a generalized linear model for positive continuous data with variance proportional to the cube of the mean."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InverseGaussianRegression(InverseGaussianRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the InverseGaussianRegression class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Dispersion` | Gets the estimated dispersion parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInverseLink(Vector<>)` | Applies the inverse link function to convert the linear predictor to the mean. |
| `ClampMu(Vector<>)` | Clamps the mean values to ensure they're positive and numerically stable. |
| `ComputeWeights(Vector<>)` | Computes the weights matrix for the iteratively reweighted least squares algorithm. |
| `ComputeWorkingResponse(Vector<>,Vector<>,Vector<>)` | Computes the working response for the iteratively reweighted least squares algorithm. |
| `CreateNewInstance` | Creates a new instance of the Inverse Gaussian Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `EstimateDispersion(Matrix<>,Vector<>)` | Estimates the dispersion parameter using Pearson residuals after model fitting. |
| `HasConverged(Vector<>,Vector<>)` | Checks if the algorithm has converged by comparing the change in coefficients. |
| `Predict(Matrix<>)` | Makes predictions for the given input data. |
| `Serialize` | Serializes the model to a byte array. |
| `Train(Matrix<>,Vector<>)` | Trains the Inverse Gaussian regression model on the provided data. |
| `ValidateInverseGaussianData(Vector<>)` | Validates that all target values are positive, as required for Inverse Gaussian regression. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dispersion` | The estimated dispersion parameter (φ = 1/λ). |
| `_options` | Configuration options for the Inverse Gaussian regression model. |

