---
title: "RidgeRegression<T>"
description: "Implements Ridge Regression (L2 regularized linear regression), which extends ordinary least squares by adding a penalty term proportional to the squared magnitude of the coefficients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Ridge Regression (L2 regularized linear regression), which extends ordinary least squares
by adding a penalty term proportional to the squared magnitude of the coefficients.

## For Beginners

Ridge Regression is a safer version of linear regression.

Regular linear regression can become unstable when:

- You have many features relative to samples
- Some features are highly correlated with each other
- The data contains noise

Ridge Regression fixes these issues by adding a "penalty" for large coefficients:

- It prevents any single feature from dominating the prediction
- It makes the model more stable and reliable
- It typically improves predictions on new, unseen data

Think of it like putting rubber bands on a flexible ruler - the bands (regularization)
keep the ruler from bending too wildly (overfitting), while still allowing it to
follow the general trend of the data.

Example usage:
```cs
var options = new RidgeRegressionOptions<double> { Alpha = 1.0 };
var ridge = new RidgeRegression<double>(options);
ridge.Train(features, targets);
var predictions = ridge.Predict(newFeatures);
```

## How It Works

Ridge Regression solves the following optimization problem:
minimize ||y - Xw||^2 + alpha * ||w||^2

This has a closed-form solution: w = (X^T X + alpha * I)^(-1) X^T y

The L2 penalty shrinks coefficients toward zero but never sets them exactly to zero,
making Ridge Regression suitable for problems where all features are expected to contribute.

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
    .ConfigureModel(new RidgeRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained RidgeRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RidgeRegression(RidgeRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `RidgeRegression` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the configuration options specific to Ridge Regression. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of Ridge Regression with the same configuration. |
| `Deserialize(Byte[])` | Deserializes a Ridge Regression model from a byte array. |
| `GetModelMetadata` | Gets metadata about the Ridge Regression model. |
| `Serialize` | Serializes the Ridge Regression model to a byte array. |
| `SolveSystemWithDecomposition(Matrix<>,Vector<>)` | Solves a linear system using the configured decomposition method. |
| `Train(Matrix<>,Vector<>)` | Trains the Ridge Regression model using the provided training data. |

