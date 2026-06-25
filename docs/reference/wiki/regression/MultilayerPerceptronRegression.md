---
title: "MultilayerPerceptronRegression"
description: "Represents a multilayer perceptron (neural network) for regression problems."
section: "Reference"
---

_Regression Models_

Represents a multilayer perceptron (neural network) for regression problems.

## For Beginners

A multilayer perceptron is like a digital brain that can learn complex patterns.

Think of it as a system of interconnected layers:

- The input layer receives your data (like house features if predicting house prices)
- The hidden layers process this information through a series of mathematical transformations
- The output layer produces the final prediction (like the predicted house price)

Each connection between neurons has a "weight" (importance) that gets adjusted as the network learns.
For example, the network might learn that square footage has a bigger impact on house prices than
the age of the house, so it assigns a larger weight to that feature.

The network improves by comparing its predictions to actual values and adjusting the weights
to reduce the difference between them.

## How It Works

The MultilayerPerceptronRegression is a neural network-based regression model that can capture complex non-linear
relationships between features and the target variable. It consists of an input layer, one or more hidden layers,
and an output layer, with each layer connected by weights and biases. The model learns by adjusting these weights
and biases through a process called backpropagation, minimizing the prediction error.

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
    .ConfigureModel(new MultilayerPerceptronRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained MultilayerPerceptronRegression.");
```

