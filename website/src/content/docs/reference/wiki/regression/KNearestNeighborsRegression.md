---
title: "KNearestNeighborsRegression<T>"
description: "Implements K-Nearest Neighbors algorithm for regression, which predicts target values by averaging the values of the K closest training examples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements K-Nearest Neighbors algorithm for regression, which predicts target values
by averaging the values of the K closest training examples.

## For Beginners

K-Nearest Neighbors is like asking your neighbors for advice.

Imagine you want to guess the price of a house:

- You look at the K most similar houses to yours (the "nearest neighbors") 
- You take the average of their prices as your prediction

The "K" is just how many neighbors you consider. If K=3, you look at the 3 most similar houses.

This approach is:

- Simple to understand: similar inputs should have similar outputs
- Makes no assumptions about the data's structure
- Works well when similar examples in your data actually have similar target values

Unlike most machine learning algorithms, KNN doesn't "learn" patterns during training - it simply
remembers all examples and does the real work at prediction time by finding similar examples.

## How It Works

K-Nearest Neighbors (KNN) is a non-parametric and instance-based learning algorithm that makes
predictions based on the similarity between the input and training samples. For regression, it computes
the average of the target values of the K nearest neighbors to the query point. The algorithm doesn't
build an explicit model but instead stores all training examples and performs computations at prediction time.

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
    .ConfigureModel(new KNearestNeighborsRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained KNearestNeighborsRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KNearestNeighborsRegression(KNearestNeighborsOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `KNearestNeighborsRegression` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | KNN is a lazy learner — no optimizer parameter injection. |
| `SoftKNNTemperature` | Gets or sets the temperature parameter for soft KNN mode. |
| `UseSoftKNN` | Gets or sets whether to use soft (differentiable) KNN mode for JIT compilation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDistance(Vector<>,Vector<>)` | Calculates the Euclidean distance between two feature vectors. |
| `Clone` | Creates a shallow copy of this KNN model including its training data. |
| `CreateInstance` | Creates a new instance of the KNearestNeighborsRegression with the same configuration as the current instance. |
| `DeepCopy` | Creates a deep copy of this KNN model including its training data. |
| `Deserialize(Byte[])` | Loads a previously serialized K-Nearest Neighbors Regression model from a byte array. |
| `GetActiveFeatureIndices` | KNN uses all features (distance-based, no coefficient pruning). |
| `GetOptions` |  |
| `OptimizeModel(Matrix<>,Vector<>)` | Optimizes the KNN model by storing the training data for later use in predictions. |
| `Predict(Matrix<>)` | Predicts target values for the provided input features using the trained KNN model. |
| `PredictSingle(Vector<>)` | Predicts the target value for a single input feature vector. |
| `Serialize` | Gets the model type of the K-Nearest Neighbors Regression model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the K-Nearest Neighbors algorithm. |
| `_xTrain` | Matrix containing the feature vectors of the training samples. |
| `_yTrain` | Vector containing the target values of the training samples. |

