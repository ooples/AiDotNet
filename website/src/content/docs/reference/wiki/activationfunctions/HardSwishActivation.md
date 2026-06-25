---
title: "HardSwishActivation<T>"
description: "Implements the Hard Swish activation function used in MobileNetV3."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Hard Swish activation function used in MobileNetV3.

## For Beginners

Hard Swish combines the benefits of ReLU and Sigmoid-like activations.

The Swish function (x * sigmoid(x)) performs well in neural networks but requires
computing the expensive sigmoid function. Hard Swish approximates this using simpler operations:

- For inputs < -3: output is 0 (similar to ReLU for negative values)
- For inputs > 3: output equals the input (similar to identity for large positive values)
- For inputs between -3 and 3: smooth transition using a simple piecewise linear function

Hard Swish is particularly important in MobileNetV3 because:

- It is faster to compute than regular Swish on mobile devices
- It provides better accuracy than ReLU for deeper networks
- It is compatible with quantized inference (8-bit integer arithmetic)

## How It Works

Hard Swish is a computationally efficient approximation of the Swish activation function.
It is defined as: f(x) = x * min(max(0, x + 3), 6) / 6

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HardSwishActivation` | Initializes a new instance of the `HardSwishActivation` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether HardSwish supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Hard Swish activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the Hard Swish activation function to each element in a tensor. |
| `Activate(Vector<>)` | Applies the Hard Swish activation function to each element in a vector. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Hard Swish function for a single input value. |
| `Derivative(Tensor<>)` | Calculates the derivative of the Hard Swish function for each element in a tensor. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the Hard Swish function for a vector input. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the HardSwish activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

