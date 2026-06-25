---
title: "MultilayerPerceptronRegression<T>"
description: "Represents a multilayer perceptron (neural network) for regression problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultilayerPerceptronRegression(MultilayerPerceptronOptions<,Matrix<>,Vector<>>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `MultilayerPerceptronRegression` class with optional custom options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Trains the neural network using the provided features and target values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivation(Vector<>,Boolean)` | Applies the activation function to an input vector. |
| `ApplyActivationDerivative(Vector<>,Boolean)` | Applies the derivative of the activation function to an input vector. |
| `ComputeGradients(Matrix<>,Vector<>)` | Computes the gradients of the loss with respect to the weights and biases. |
| `ComputeLoss(Vector<>,Vector<>)` | Computes the mean squared error loss between predictions and targets. |
| `ComputeOutputLayerDelta(Vector<>,Vector<>,Vector<>)` | Computes the error delta for the output layer during backpropagation. |
| `CreateInstance` | Creates a new instance of the `MultilayerPerceptronRegression` class with the same options and regularization as this instance. |
| `Deserialize(Byte[])` | Deserializes the neural network model from a byte array. |
| `ForwardPass(Vector<>)` | Performs a forward pass through the neural network for a single input vector. |
| `InitializeNetwork` | Initializes the neural network structure with random weights. |
| `OptimizeModel(Matrix<>,Vector<>)` | Gets the type of regression model. |
| `Predict(Matrix<>)` | Generates predictions for new data points using the trained neural network. |
| `Serialize` | Serializes the neural network model to a byte array for storage or transmission. |
| `UpdateParameters(List<Matrix<>>,List<Vector<>>,Int32)` | Updates the weights and biases based on the computed gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | The bias values for each layer of the neural network. |
| `_optimizer` | The optimization algorithm used to update the weights and biases during training. |
| `_options` | The configuration options for the multilayer perceptron. |
| `_weights` | The weights connecting the layers of the neural network. |

