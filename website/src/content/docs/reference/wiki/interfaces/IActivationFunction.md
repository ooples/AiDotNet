---
title: "IActivationFunction<T>"
description: "Defines an interface for activation functions used in neural networks and other machine learning algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for activation functions used in neural networks and other machine learning algorithms.

## How It Works

**For Beginners:** An activation function is like a decision-maker in a neural network.

Imagine each neuron (node) in a neural network receives a number as input. The activation 
function decides how strongly that neuron should "fire" or activate based on that input.

For example:

- If the input is very negative, the neuron might not activate at all (output = 0)
- If the input is very positive, the neuron might activate fully (output = 1)
- If the input is around zero, the neuron might activate partially

Different activation functions create different patterns of activation, which helps
neural networks learn different types of patterns in data. Common activation functions
include Sigmoid, ReLU (Rectified Linear Unit), and Tanh (Hyperbolic Tangent).

This interface defines the standard methods that all activation functions must implement.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether this activation function supports GPU-resident training. |
| `SupportsJitCompilation` | Gets whether this activation function supports JIT compilation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the activation function to the input value. |
| `Activate(Tensor<>)` | Applies the activation function to each element in a tensor. |
| `Activate(Vector<>)` | Applies the activation function to each element in a vector. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative (slope) of the activation function at the given input value. |
| `Derivative(Tensor<>)` | Calculates the derivative for each element in a tensor. |
| `Derivative(Vector<>)` | Calculates the derivative matrix for a vector input. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the activation function on GPU. |

