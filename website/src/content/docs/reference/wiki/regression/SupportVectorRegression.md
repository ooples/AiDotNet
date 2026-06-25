---
title: "SupportVectorRegression<T>"
description: "Implements Support Vector Regression (SVR), which creates a regression model by finding a hyperplane that lies within a specified margin (epsilon) of the training data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Support Vector Regression (SVR), which creates a regression model by finding
a hyperplane that lies within a specified margin (epsilon) of the training data.
This approach is effective for both linear and nonlinear regression problems.

## For Beginners

Support Vector Regression is like creating a tunnel through your data.

Think of it like this:

- You want to draw a line (or curve) through your data points
- Instead of drawing the line directly through the points, you create a tunnel of a certain width
- You try to include as many points as possible inside this tunnel
- Points outside the tunnel are called "support vectors" and help define its shape

For example, when predicting house prices, SVR would create a tunnel through the data
that captures the general trend while allowing some houses to fall outside the tunnel
if they're unusually priced for their features.

## How It Works

Support Vector Regression (SVR) works by:

- Transforming data into a higher-dimensional space using kernel functions
- Finding the optimal hyperplane that fits within an epsilon-width tube around the data
- Using only a subset of the training examples (support vectors) to make predictions
- Balancing model complexity and training error through the C parameter

Unlike traditional regression methods that minimize squared errors, SVR aims to find
a function that deviates from training data by no more than epsilon while remaining as flat as possible.

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
    .ConfigureModel(new SupportVectorRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained SupportVectorRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SupportVectorRegression(SupportVectorRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Creates a new Support Vector Regression model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | SVR uses SMO algorithm — random parameter injection is not applicable. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Serializes the support vector regression model to a byte array for storage or transmission. |
| `ComputeBounds(,,,)` | Computes the bounds for alpha coefficient optimization. |
| `CreateInstance` | Creates a new instance of the Support Vector Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the support vector regression model from a byte array. |
| `GetActiveFeatureIndices` | Optimizes the SVR model using the provided input data and target values. |
| `GetModelMetadata` | Gets metadata about the SVR model. |
| `GetOptions` |  |
| `Predict(Matrix<>)` | Predicts target values for a matrix of input features. |
| `PredictSingle(Vector<>)` | Predicts a target value for a single input feature vector. |
| `SequentialMinimalOptimization(Matrix<>,Vector<>)` | Implements the Sequential Minimal Optimization (SMO) algorithm for SVR. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Support Vector Regression model. |
| `_random` | Selects a second alpha coefficient to optimize along with the first one. |

