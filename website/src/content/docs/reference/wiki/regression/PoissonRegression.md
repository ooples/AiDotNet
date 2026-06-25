---
title: "PoissonRegression<T>"
description: "Implements Poisson regression, a generalized linear model used for modeling count data and contingency tables."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Poisson regression, a generalized linear model used for modeling count data and contingency tables.

## How It Works

Poisson regression is appropriate when the dependent variable represents counts, such as the number of occurrences
of an event in a fixed period of time or space. It assumes that the response variable follows a Poisson distribution
and uses a log link function to relate the expected value of the response to the linear predictor.

The model is fitted using iteratively reweighted least squares (IRLS), a form of maximum likelihood estimation.

For Beginners:
Poisson regression is used when you're trying to predict counts (like number of customer visits, number of accidents,
etc.). Unlike linear regression, it ensures predictions are always non-negative and can handle cases where the
variance increases with the mean, which is common in count data.

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PoissonRegression(PoissonRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the PoissonRegression class with the specified options and regularization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeWeights(Vector<>)` | Computes the weights matrix for the iteratively reweighted least squares algorithm. |
| `ComputeWorkingResponse(Matrix<>,Vector<>,Vector<>,Vector<>)` | Computes the working response for the iteratively reweighted least squares algorithm. |
| `CreateNewInstance` | Creates a new instance of the Poisson Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `HasConverged(Vector<>,Vector<>)` | Checks if the algorithm has converged by comparing the change in coefficients. |
| `Predict(Matrix<>)` | Makes predictions for the given input data. |
| `PredictMean(Matrix<>,Vector<>)` | Predicts the mean (expected value) for the given input data using the current model parameters. |
| `Serialize` | Serializes the model to a byte array. |
| `Train(Matrix<>,Vector<>)` | Trains the Poisson regression model on the provided data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `MinSafeValue` | Minimum value for numerical stability guards to prevent division by zero. |
| `_options` | Configuration options for the Poisson regression model. |
| `_yShift` | Shift applied to y to make it positive (0 if y was already positive). |

