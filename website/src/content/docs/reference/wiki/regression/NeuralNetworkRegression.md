---
title: "NeuralNetworkRegression<T>"
description: "A neural network regression model that can learn complex non-linear relationships in data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

A neural network regression model that can learn complex non-linear relationships in data.

## How It Works

This class implements a fully connected feedforward neural network for regression tasks.
It supports multiple hidden layers with customizable activation functions and uses
gradient-based optimization to learn from data.

The neural network architecture is defined by specifying the number of neurons in each layer,
with the first layer corresponding to the input features and the last layer to the output.

For Beginners:
A neural network is a machine learning model inspired by the human brain. It consists of layers
of interconnected "neurons" that process input data to make predictions. Each connection has a
"weight" that determines its importance, and these weights are adjusted during training to improve
the model's accuracy. This process is similar to how we learn from experience.

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralNetworkRegression(NeuralNetworkRegressionOptions<,Matrix<>,Vector<>>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the NeuralNetworkRegression class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Makes predictions for the given input data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateGradients(List<Vector<>>,List<Vector<>>,List<Matrix<>>,List<Vector<>>)` | Accumulates gradients for the weights and biases based on the activations and deltas. |
| `ApplyActivation(Vector<>,Boolean)` | Applies the appropriate activation function to the input vector. |
| `ApplyActivationDerivative(Vector<>,Boolean)` | Applies the derivative of the appropriate activation function to the input vector. |
| `BackwardPass(List<Vector<>>,Vector<>)` | Performs a backward pass through the neural network to compute the gradients. |
| `CreateInstance` | Creates a new instance of the Neural Network Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `ForwardPass(Vector<>)` | Performs a forward pass through the neural network. |
| `GetBatchElements(Vector<>,Int32[],Int32,Int32)` | Extracts a batch of elements from a vector based on the provided indices. |
| `GetBatchRows(Matrix<>,Int32[],Int32,Int32)` | Extracts a batch of rows from a matrix based on the provided indices. |
| `GetOptions` |  |
| `InitializeNetwork` | Initializes the neural network by creating weight matrices and bias vectors for each layer. |
| `OptimizeModel(Matrix<>,Vector<>)` | Gets the type of the model. |
| `Serialize` | Serializes the model to a byte array. |
| `ShuffleArray(Int32[])` | Randomly shuffles an array using the Fisher-Yates algorithm. |
| `Train(Matrix<>,Vector<>)` | Trains the neural network on the provided data. |
| `UpdateParameters(List<Matrix<>>,List<Vector<>>,Int32)` | Updates the network parameters (weights and biases) using the accumulated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | The bias vectors for each layer of the neural network. |
| `_optimizer` | The optimization algorithm used to update the model parameters during training. |
| `_options` | Configuration options for the neural network regression model. |
| `_weights` | The weight matrices for each layer of the neural network. |

