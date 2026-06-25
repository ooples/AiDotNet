---
title: "NeuralNetworkRegression"
description: "A neural network regression model that can learn complex non-linear relationships in data."
section: "Reference"
---

_Regression Models_

A neural network regression model that can learn complex non-linear relationships in data.

## How It Works

This class implements a fully connected feedforward neural network for regression tasks. It supports multiple hidden layers with customizable activation functions and uses gradient-based optimization to learn from data. 

The neural network architecture is defined by specifying the number of neurons in each layer, with the first layer corresponding to the input features and the last layer to the output. 

For Beginners: A neural network is a machine learning model inspired by the human brain. It consists of layers of interconnected "neurons" that process input data to make predictions. Each connection has a "weight" that determines its importance, and these weights are adjusted during training to improve the model's accuracy. This process is similar to how we learn from experience.

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
    .ConfigureModel(new NeuralNetworkRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained NeuralNetworkRegression.");
```

