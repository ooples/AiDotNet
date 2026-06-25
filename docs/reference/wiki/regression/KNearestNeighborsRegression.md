---
title: "KNearestNeighborsRegression"
description: "Implements K-Nearest Neighbors algorithm for regression, which predicts target values by averaging the values of the K closest training examples."
section: "Reference"
---

_Regression Models_

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

