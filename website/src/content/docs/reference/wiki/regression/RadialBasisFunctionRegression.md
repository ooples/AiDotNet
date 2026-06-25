---
title: "RadialBasisFunctionRegression<T>"
description: "Implements Radial Basis Function (RBF) Regression, a technique that uses radial basis functions as the basis for approximating complex nonlinear relationships between inputs and outputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Radial Basis Function (RBF) Regression, a technique that uses radial basis functions
as the basis for approximating complex nonlinear relationships between inputs and outputs.

## For Beginners

Think of RBF regression as placing a set of "bell curves" at strategic locations in your input space.
Each curve gives a strong response when an input is close to its center and a weak response when it's far away.
The model predicts by combining these responses with learned weights. This approach is particularly good at
modeling complex, non-linear relationships in data.

## How It Works

Radial Basis Function Regression works by transforming the input space using a set of radial basis functions,
each centered at a different point. These functions produce a response that depends on the distance from the
input to the center point. The model then combines these responses linearly to make predictions.

The algorithm first selects a set of centers (typically using k-means clustering), computes the RBF features
for each input point, and then solves a linear regression problem to find the optimal weights.

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
    .ConfigureModel(new RadialBasisFunctionRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained RadialBasisFunctionRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RadialBasisFunctionRegression(RadialBasisFunctionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the RadialBasisFunctionRegression class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Optimizes the model parameters based on the training data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Deep copy via serialization to preserve private _centers and _weights. |
| `ComputeRBFFeatures(Matrix<>)` | Computes the RBF features for a matrix of input points. |
| `ComputeRBFFeaturesSingle(Vector<>)` | Computes the RBF features for a single input vector. |
| `CreateInstance` | Creates a new instance of the radial basis function regression model with the same options. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `GetActiveFeatureIndices` | Returns all features since RBF uses distance-based kernels across all features. |
| `GetOptions` |  |
| `Predict(Matrix<>)` | Makes predictions for the given input data. |
| `PredictSingle(Vector<>)` | Predicts the value for a single input vector. |
| `RbfKernel()` | Calculates the Euclidean distance between two vectors. |
| `SelectCenters(Matrix<>)` | Selects centers for the radial basis functions using k-means clustering. |
| `Serialize` | Serializes the model to a byte array. |
| `SolveLinearRegression(Matrix<>,Vector<>)` | Solves a linear regression problem to find the optimal weights using ridge regularization. |

## Fields

| Field | Summary |
|:-----|:--------|
| `MinimumStabilityLambda` | Absolute floor for the ridge penalty when diagonal magnitude is near zero. |
| `StabilityLambdaScale` | Relative scale factor for the adaptive ridge penalty (fraction of mean diagonal magnitude). |
| `_centers` | The centers of the radial basis functions. |
| `_options` | Configuration options for the radial basis function regression model. |
| `_weights` | The weights used to combine the radial basis function outputs. |

