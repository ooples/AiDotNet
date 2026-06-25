---
title: "ELUActivation<T>"
description: "Implements the Exponential Linear Unit (ELU) activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Exponential Linear Unit (ELU) activation function for neural networks.

## For Beginners

ELU is an activation function that, like ReLU, returns the input directly for positive values.
For negative inputs, it returns a smooth curve that approaches -alpha as the input becomes more negative.

Key advantages of ELU include:

- It helps prevent "dying neurons" (a problem with ReLU) by allowing negative values
- It has a smooth curve for negative inputs, which can help with gradient-based learning
- The parameter alpha controls how negative the curve can go
- It centers the activations closer to zero, which can speed up learning

ELU is often used in deep neural networks where ReLU might cause training issues.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ELUActivation(Double)` | Initializes a new instance of the ELUActivation class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the alpha parameter that controls the saturation value for negative inputs. |
| `SupportsGpuTraining` | Gets whether ELU supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the ELU activation function to a scalar input value. |
| `Activate(Tensor<>)` | Applies ELU to a tensor via the engine so the gradient tape records the op. |
| `Activate(Vector<>)` | Applies the ELU activation function to each element of an input vector. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the ELU activation function for a scalar input value. |
| `Derivative(Vector<>)` | Calculates the derivative of the ELU activation function for each element of an input vector. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the ELU activation function on GPU. |
| `SupportsScalarOperations` | Determines if the activation function supports operations on individual scalar values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The alpha parameter that controls the saturation value for negative inputs. |

