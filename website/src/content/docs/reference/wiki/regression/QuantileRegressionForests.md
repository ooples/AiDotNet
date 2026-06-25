---
title: "QuantileRegressionForests<T>"
description: "Implements Quantile Regression Forests, an extension of Random Forests that can predict conditional quantiles of the target variable, not just the conditional mean."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Quantile Regression Forests, an extension of Random Forests that can predict conditional quantiles
of the target variable, not just the conditional mean.

## For Beginners

While standard Random Forests tell you the average prediction, Quantile Regression Forests can tell you about
the entire range of possible outcomes. For example, they can predict not just the expected value, but also the
10th percentile (a pessimistic scenario) or the 90th percentile (an optimistic scenario). This is particularly
useful when you need to understand the uncertainty in your predictions or when the relationship between variables
varies across different parts of the distribution.

## How It Works

Quantile Regression Forests extend the Random Forests algorithm to estimate the full conditional distribution
of the response variable, not just its mean. This allows for prediction of any quantile of the response variable,
providing a more complete picture of the relationship between predictors and the response.

The algorithm works by building multiple decision trees on bootstrap samples of the training data, similar to
Random Forests. However, instead of averaging the predictions, it uses the empirical distribution of the predictions
from all trees to estimate quantiles.

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
    .ConfigureModel(new QuantileRegressionForests<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained QuantileRegressionForests.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuantileRegressionForests` | Initializes a new instance with default settings. |
| `QuantileRegressionForests(QuantileRegressionForestsOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the QuantileRegressionForests class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDepth` | Gets the maximum depth of the trees in the forest. |
| `NumberOfTrees` | Gets the number of trees in the forest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportancesAsync(Int32)` | Asynchronously calculates the importance of each feature in the model. |
| `CreateNewInstance` | Creates a new instance of the Quantile Regression Forests model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `PredictAsync(Matrix<>)` | Asynchronously makes predictions for the given input data. |
| `PredictQuantileAsync(Matrix<>,Double)` | Asynchronously predicts a specific quantile of the target variable for the given input data. |
| `SampleWithReplacement(Matrix<>,Vector<>)` | Creates a bootstrap sample of the training data by sampling with replacement. |
| `Serialize` | Serializes the model to a byte array. |
| `TrainAsync(Matrix<>,Vector<>)` | Asynchronously trains the Quantile Regression Forests model on the provided data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Quantile Regression Forests model. |
| `_random` | Random number generator used for bootstrap sampling. |
| `_trees` | The collection of decision trees that make up the forest. |

