---
title: "LassoRegression"
description: "Implements Lasso Regression (L1 regularized linear regression), which extends ordinary least squares by adding a penalty term proportional to the absolute magnitude of the coefficients."
section: "Reference"
---

_Regression Models_

Implements Lasso Regression (L1 regularized linear regression), which extends ordinary least squares
by adding a penalty term proportional to the absolute magnitude of the coefficients.

## For Beginners

Lasso Regression automatically selects important features.

While Ridge Regression keeps all features but shrinks their coefficients,
Lasso can completely eliminate unimportant features by setting their
coefficients to zero. This is useful when:

- You have many features and want to identify the most important ones
- You want a simpler, more interpretable model
- You suspect only a few features actually matter

Example usage:
```cs
var options = new LassoRegressionOptions<double> { Alpha = 1.0 };
var lasso = new LassoRegression<double>(options);
lasso.Train(features, targets);
var predictions = lasso.Predict(newFeatures);

// Check which features were selected (non-zero coefficients)
var selectedFeatures = lasso.GetActiveFeatureIndices();
```

## How It Works

Lasso Regression solves the following optimization problem:
minimize (1/2n) * ||y - Xw||^2 + alpha * ||w||_1

Unlike Ridge Regression, Lasso uses coordinate descent optimization because
the L1 penalty is not differentiable at zero.

The L1 penalty can shrink coefficients exactly to zero, making Lasso
useful for automatic feature selection.

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
    .ConfigureModel(new LassoRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained LassoRegression.");
```

