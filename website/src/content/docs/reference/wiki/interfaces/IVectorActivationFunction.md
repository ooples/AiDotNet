---
title: "IVectorActivationFunction<T>"
description: "Defines activation functions that operate on vectors and tensors in neural networks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines activation functions that operate on vectors and tensors in neural networks.

## How It Works

Activation functions introduce non-linearity into neural networks, allowing them to learn
complex patterns in data. This interface provides methods to apply activation functions
to vectors and tensors, as well as calculate their derivatives for backpropagation.

**For Beginners:** Activation functions are like "decision makers" in neural networks.

Imagine you're deciding whether to go outside based on the temperature:

- If it's below 60—F, you definitely won't go (output = 0)
- If it's above 75—F, you definitely will go (output = 1)
- If it's between 60-75—F, you're somewhat likely to go (output between 0 and 1)

This is similar to how activation functions work. They take the input from previous
calculations in the neural network and transform it into an output that determines
how strongly a neuron "fires" or activates. Without activation functions, neural
networks would just be doing simple linear calculations and couldn't learn complex patterns.

Common activation functions include:

- Sigmoid: Outputs values between 0 and 1 (like our temperature example)
- ReLU: Outputs the input if positive, or zero if negative
- Tanh: Outputs values between -1 and 1

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsJitCompilation` | Gets whether this activation function supports JIT compilation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies the activation function to each element in a tensor. |
| `Activate(Vector<>)` | Applies the activation function to each element in a vector. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Backward(Tensor<>,Tensor<>)` | Calculates the backward pass gradient for this activation function. |
| `Derivative(Tensor<>)` | Calculates the derivative of the activation function for each element in a tensor. |
| `Derivative(Vector<>)` | Calculates the derivative of the activation function for each element in a vector. |

