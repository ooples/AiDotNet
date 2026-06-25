---
title: "NeuralNetworkHelper<T>"
description: "Provides helper methods for neural network operations including activation functions and loss functions."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides helper methods for neural network operations including activation functions and loss functions.

## For Beginners

Neural networks are computing systems inspired by the human brain. They process information
through interconnected nodes (neurons) that transform input data using mathematical functions.
This helper class provides those mathematical functions needed to build neural networks.

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivation(Tensor<>,IActivationFunction<>,IVectorActivationFunction<>)` | Applies an activation function to a tensor of values. |
| `ApplyActivation(Vector<>,IActivationFunction<>,IVectorActivationFunction<>)` | Applies an activation function to a vector of values. |
| `ApplyOutputActivation(Tensor<>,NeuralNetworkArchitecture<>)` | Applies the appropriate activation function to the output tensor based on the task type. |
| `ApplySigmoid(Tensor<>)` | Applies the sigmoid activation function to all elements in a tensor. |
| `ApplySoftmax(Tensor<>)` | Applies the softmax activation function to a tensor along the last dimension. |
| `ApplyTanh(Tensor<>)` | Applies the tanh activation function to all elements in a tensor. |
| `EuclideanDistance(Vector<>,Vector<>)` | Calculates the Euclidean distance between two vectors. |
| `GetDefaultActivationFunction(NeuralNetworkTaskType)` | Gets the default activation function based on the task type. |
| `GetDefaultLossFunction(NeuralNetworkTaskType)` | Gets the default loss function based on the task type. |
| `GetDefaultVectorActivationFunction(NeuralNetworkTaskType)` | Gets the default vector activation function based on the task type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Provides operations for the numeric type T. |

