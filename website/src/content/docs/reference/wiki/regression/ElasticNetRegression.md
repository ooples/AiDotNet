---
title: "ElasticNetRegression<T>"
description: "Implements Elastic Net Regression (combined L1 and L2 regularized linear regression), which extends ordinary least squares by adding both L1 (Lasso) and L2 (Ridge) penalty terms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Elastic Net Regression (combined L1 and L2 regularized linear regression),
which extends ordinary least squares by adding both L1 (Lasso) and L2 (Ridge) penalty terms.

## For Beginners

Elastic Net gives you the best of both worlds.

Lasso is great at selecting important features, but when features are correlated,
it tends to arbitrarily pick one and zero out the others. Ridge handles correlated
features well but doesn't do feature selection.

Elastic Net solves both problems:

- It can still set coefficients to zero (like Lasso) for feature selection
- It groups correlated features together (like Ridge) instead of picking arbitrarily

Example usage:
```cs
var options = new ElasticNetRegressionOptions<double> { Alpha = 1.0, L1Ratio = 0.5 };
var elasticNet = new ElasticNetRegression<double>(options);
elasticNet.Train(features, targets);
var predictions = elasticNet.Predict(newFeatures);

// Check which features were selected (non-zero coefficients)
var selectedFeatures = elasticNet.GetActiveFeatureIndices();
```

## How It Works

Elastic Net Regression solves the following optimization problem:
minimize (1/2n) * ||y - Xw||^2 + alpha * l1_ratio * ||w||_1 + alpha * (1 - l1_ratio) * ||w||^2 / 2

Like Lasso, Elastic Net uses coordinate descent optimization because
the L1 penalty is not differentiable at zero.

Elastic Net combines the benefits of both Ridge and Lasso:

- Feature selection from L1 (can set coefficients to exactly zero)
- Stability with correlated features from L2 (groups correlated features together)

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
    .ConfigureModel(new ElasticNetRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained ElasticNetRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElasticNetRegression(ElasticNetRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `ElasticNetRegression` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IterationsUsed` | Gets the number of iterations used in the last training. |
| `NumberOfSelectedFeatures` | Gets the number of non-zero coefficients (selected features). |
| `Options` | Gets the configuration options specific to Elastic Net Regression. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of Elastic Net Regression with the same configuration. |
| `Deserialize(Byte[])` | Deserializes an Elastic Net Regression model from a byte array. |
| `ElasticNetSoftThreshold(,,,,)` | Applies the elastic net soft-thresholding operator for combined L1 and L2 regularization. |
| `GetModelMetadata` | Gets metadata about the Elastic Net Regression model. |
| `Serialize` | Serializes the Elastic Net Regression model to a byte array. |
| `Train(Matrix<>,Vector<>)` | Trains the Elastic Net Regression model using coordinate descent optimization. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_iterationsUsed` | Stores the number of iterations used in the last training. |

