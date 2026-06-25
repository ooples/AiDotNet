---
title: "RandomForestRegression<T>"
description: "Implements Random Forest Regression, an ensemble learning method that operates by constructing multiple decision trees during training and outputting the average prediction of the individual trees."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Random Forest Regression, an ensemble learning method that operates by constructing multiple
decision trees during training and outputting the average prediction of the individual trees.

## For Beginners

Think of Random Forest as a committee of decision trees, where each tree votes on the prediction.
By combining many trees, each trained slightly differently, the model becomes more robust and accurate
than any single tree. It's like asking multiple experts for their opinion and taking the average.

## How It Works

Random Forest Regression combines multiple decision trees to improve prediction accuracy and control overfitting.
Each tree is trained on a bootstrap sample of the training data, and at each node, only a random subset of
features is considered for splitting. The final prediction is the average of predictions from all trees.

The algorithm's key strengths include robustness to outliers, good performance on high-dimensional data,
and the ability to capture non-linear relationships without requiring extensive hyperparameter tuning.

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
    .ConfigureModel(new RandomForestRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained RandomForestRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomForestRegression` | Initializes a new instance with default settings. |
| `RandomForestRegression(RandomForestRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the RandomForestRegression class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDepth` | Gets the maximum depth of the trees in the forest. |
| `NumberOfTrees` | Gets the number of trees in the forest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportancesAsync(Int32)` | Asynchronously calculates the importance of each feature in the model. |
| `Clone` | Deep copy via serialization to preserve the private _trees list. |
| `CreateNewInstance` | Creates a new instance of the Random Forest regression model with the same options. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `GetBootstrapSampleIndices(Int32)` | Generates indices for a bootstrap sample of the training data. |
| `GetModelMetadata` | Gets metadata about the model. |
| `PredictAsync(Matrix<>)` | Asynchronously makes predictions for the given input data. |
| `Serialize` | Serializes the model to a byte array. |
| `TrainAsync(Matrix<>,Vector<>)` | Asynchronously trains the Random Forest regression model on the provided data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Random Forest regression model. |
| `_random` | Random number generator used for bootstrap sampling and feature selection. |
| `_trees` | The collection of decision trees that make up the forest. |

