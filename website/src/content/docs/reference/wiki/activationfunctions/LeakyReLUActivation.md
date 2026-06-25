---
title: "LeakyReLUActivation<T>"
description: "Implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function for neural networks.

## For Beginners

The Leaky ReLU activation function is a variation of the standard ReLU function.

How it works:

- For positive inputs (x > 0): It returns the input unchanged (like a straight line)
- For negative inputs (x = 0): It returns a small fraction of the input (a * x)

The main advantage of Leaky ReLU over standard ReLU is that it never completely "turns off" 
neurons for negative inputs. Instead, it allows a small gradient to flow through, which helps
prevent the "dying ReLU" problem where neurons can stop learning during training.

Think of it like a water pipe that:

- Allows full flow when the input is positive
- Allows a small "leak" when the input is negative (controlled by the alpha parameter)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeakyReLUActivation` | Initializes a new instance of the Leaky ReLU activation function with the default slope (alpha = 0.01). |
| `LeakyReLUActivation(Double)` | Initializes a new instance of the Leaky ReLU activation function with the specified alpha parameter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the slope coefficient for negative input values. |
| `SupportsGpuTraining` | Gets whether LeakyReLU supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Leaky ReLU activation function to a single value. |
| `Activate(Tensor<>)` | Applies the Leaky ReLU activation function to each element in a tensor. |
| `Activate(Vector<>)` | Applies the Leaky ReLU activation function to a vector of values. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative (gradient) of the Leaky ReLU function for a single value. |
| `Derivative(Tensor<>)` | Calculates the derivative of the Leaky ReLU function for each element in a tensor. |
| `Derivative(Vector<>)` | Calculates the derivative (gradient) of the Leaky ReLU function for a vector of values. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the Leaky ReLU activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function can operate on individual scalar values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The slope coefficient for negative input values. |

