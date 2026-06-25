---
title: "LocallyWeightedRegression<T>"
description: "Implements Locally Weighted Regression, a non-parametric approach that creates a different model for each prediction point based on the weighted influence of nearby training examples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Locally Weighted Regression, a non-parametric approach that creates a different model
for each prediction point based on the weighted influence of nearby training examples.

## For Beginners

Locally Weighted Regression is like having a personalized prediction for each point.

Instead of creating a single model for all data (like linear regression does), LWR:

- Creates a new, custom model for each prediction point you want to estimate
- Gives more importance to training examples that are close to your prediction point
- Gives less importance to training examples that are far away

Imagine predicting house prices: When estimating the price of a specific house, LWR would:

- Give most influence to similar houses in the same neighborhood
- Give moderate influence to somewhat similar houses in nearby areas
- Give little or no influence to very different houses in distant locations

This approach is flexible and works well for complex patterns, but requires keeping all training
data around for making predictions, which can be computationally intensive for large datasets.

## How It Works

Locally Weighted Regression (LWR) is a memory-based, non-parametric method that creates a unique model
for each prediction point. Unlike global regression methods that find a single model for all data,
LWR fits a separate weighted regression model for each query point, giving higher influence to
nearby training examples. This approach provides excellent flexibility for modeling complex, nonlinear
relationships without specifying a fixed functional form.

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
    .ConfigureModel(new LocallyWeightedRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained LocallyWeightedRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LocallyWeightedRegression(LocallyWeightedRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `LocallyWeightedRegression` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Stores the training data for later use in making predictions. |
| `UseSoftMode` | Gets or sets whether to use soft (differentiable) mode for JIT compilation support. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Deep copy via serialization. |
| `ComputeWeights(Vector<>)` | Computes weights for each training example based on their distance to the input point. |
| `CreateInstance` | Creates a new instance of the LocallyWeightedRegression with the same configuration as the current instance. |
| `Deserialize(Byte[])` | Loads a previously serialized Locally Weighted Regression model from a byte array. |
| `GetActiveFeatureIndices` | Returns all features used during training. |
| `KernelFunction()` | Applies a kernel function to transform distances into weights. |
| `Predict(Matrix<>)` | Predicts target values for the provided input features using the trained Locally Weighted Regression model. |
| `PredictSingle(Vector<>)` | Predicts the target value for a single input feature vector. |
| `Serialize` | Gets the model type of the Locally Weighted Regression model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `MinimumStabilityStrength` | Absolute floor for the ridge penalty when diagonal magnitude is near zero. |
| `StabilityStrengthScale` | Relative scale factor for the adaptive ridge penalty (fraction of mean diagonal magnitude). |
| `ZeroWeightTolerance` | Tolerance below which total kernel weight is treated as zero (no neighbors in bandwidth). |
| `_options` | Configuration options for the Locally Weighted Regression algorithm. |
| `_xTrain` | Matrix containing the feature vectors of the training samples. |
| `_yTrain` | Vector containing the target values of the training samples. |

