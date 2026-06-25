---
title: "GradientBoostingRegression<T>"
description: "Implements a Gradient Boosting Regression model, which combines multiple decision trees sequentially to create a powerful ensemble that learns from the errors of previous trees."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements a Gradient Boosting Regression model, which combines multiple decision trees
sequentially to create a powerful ensemble that learns from the errors of previous trees.

## For Beginners

Gradient Boosting is like having a team of experts who learn from each other's mistakes.

Imagine you're trying to predict house prices:

- You start with a simple guess (the average price of all houses)
- You build a decision tree to predict where your guess was wrong
- You adjust your prediction a little bit based on this tree
- You build another tree to predict where you're still making mistakes
- You keep adding trees, each one focusing on fixing the remaining errors

The "gradient" part refers to how it identifies mistakes, and "boosting" means it builds trees 
sequentially, with each tree boosting the performance of the ensemble.

This approach is very powerful because:

- It learns complex patterns gradually
- It focuses its effort on the hard-to-predict cases
- It combines many simple models (trees) into a strong predictive model

## How It Works

Gradient Boosting is an ensemble technique that builds decision trees sequentially, with each tree
correcting the errors made by the previous trees. The model starts with a simple prediction (typically
the mean of the target values) and iteratively adds trees that predict the residuals (errors) of the
current ensemble. These predictions are added to the ensemble with a learning rate that controls the
contribution of each tree, helping to prevent overfitting.

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
    .ConfigureModel(new GradientBoostingRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained GradientBoostingRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientBoostingRegression(GradientBoostingRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `GradientBoostingRegression` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfTrees` | Gets the number of trees in the ensemble model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportancesAsync(Int32)` | Calculates the importance scores for all features used in the model. |
| `Clone` |  |
| `CreateNewInstance` | Creates a new instance of the gradient boosting regression model with the same configuration. |
| `Deserialize(Byte[])` | Loads a previously serialized Gradient Boosting Regression model from a byte array. |
| `GetActiveFeatureIndices` |  |
| `GetModelMetadata` | Gets metadata about the Gradient Boosting Regression model and its configuration. |
| `GetOptions` |  |
| `PredictAsync(Matrix<>)` | Asynchronously predicts target values for the provided input features using the trained Gradient Boosting model. |
| `Serialize` | Serializes the Gradient Boosting Regression model to a byte array for storage or transmission. |
| `TrainAsync(Matrix<>,Vector<>)` | Asynchronously trains the Gradient Boosting Regression model using the provided input features and target values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_initialPrediction` | The initial prediction value, typically the mean of the target values. |
| `_options` | Configuration options for the Gradient Boosting algorithm. |
| `_trees` | Collection of decision trees that make up the ensemble. |

