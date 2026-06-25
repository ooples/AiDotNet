---
title: "GammaRegression<T>"
description: "Implements Gamma regression, a generalized linear model for positive continuous data with right-skewed distributions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GammaRegression(GammaRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the GammaRegression class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Dispersion` | Gets the estimated dispersion parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInverseLink(Vector<>)` | Applies the inverse link function to convert the linear predictor to the mean. |
| `ApplyLink(Vector<>)` | Applies the link function to convert the mean to the linear predictor scale. |
| `ClampMu(Vector<>)` | Clamps the mean values to ensure they're positive and numerically stable. |
| `ComputeWeights(Vector<>)` | Computes the weights matrix for the iteratively reweighted least squares algorithm. |
| `ComputeWorkingResponse(Vector<>,Vector<>,Vector<>)` | Computes the working response for the iteratively reweighted least squares algorithm. |
| `CreateNewInstance` | Creates a new instance of the Gamma Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `EstimateDispersion(Matrix<>,Vector<>)` | Estimates the dispersion parameter using Pearson residuals after model fitting. |
| `HasConverged(Vector<>,Vector<>)` | Checks if the algorithm has converged by comparing the change in coefficients. |
| `Predict(Matrix<>)` | Makes predictions for the given input data. |
| `Serialize` | Serializes the model to a byte array. |
| `Train(Matrix<>,Vector<>)` | Trains the Gamma regression model on the provided data. |
| `ValidateGammaData(Vector<>)` | Validates that all target values are positive, as required for Gamma regression. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dispersion` | The estimated dispersion parameter (φ = 1/shape). |
| `_options` | Configuration options for the Gamma regression model. |

