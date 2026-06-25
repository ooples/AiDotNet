---
title: "NeuralNetworkKernel<T>"
description: "Implements the Neural Network (Arc-Cosine) kernel that corresponds to infinitely wide neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Neural Network (Arc-Cosine) kernel that corresponds to infinitely wide neural networks.

## For Beginners

The Neural Network kernel (also called Arc-Cosine kernel) is fascinating
because it exactly corresponds to the behavior of infinitely wide neural networks with
specific activation functions.

Key insight: As a neural network gets wider and wider (more neurons in each layer),
its behavior becomes more predictable and can be described by a Gaussian Process with
this kernel. This connection is called the "Neural Network Gaussian Process" (NNGP).

The kernel depends on the activation function:

- Order 0: Step function (Heaviside) - Measures angle between inputs
- Order 1: ReLU - Captures piecewise linear behavior
- Order 2: ReQU (Rectified Quadratic Unit) - Smoother, more expressive

## How It Works

Mathematical form:
k_n(x, x') = (1/π) × ||x||^n × ||x'||^n × J_n(θ)

Where:

- θ = angle between x and x' (arccos of normalized dot product)
- J_n(θ) = (-1)^n × (sin(θ))^(2n+1) × (∂/∂t)^n [(π-θ)/sin(θ)] evaluated at t=cos(θ)

For n=1 (ReLU): J_1(θ) = sin(θ) + (π-θ)cos(θ)

Why use this kernel?

1. **Neural Network Connection**: Analyze what infinite neural networks can learn
2. **Deep Architectures**: Can be composed for "deep" kernel behavior
3. **Non-stationary**: Unlike RBF, it's not translation-invariant
4. **Theoretical Insights**: Helps understand deep learning through GP lens

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralNetworkKernel(Int32,Double,Double)` | Initializes a new Neural Network kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BiasVariance` | Gets the bias variance. |
| `Order` | Gets the kernel order. |
| `WeightVariance` | Gets the weight variance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Neural Network kernel value between two vectors. |
| `ComputeArcCosineFunction(Double,Double)` | Computes the arc-cosine function J_n(θ). |
| `ToDeep(Int32)` | Creates a "deep" version of the kernel by composing multiple layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biasVariance` | The bias variance parameter. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_order` | The order of the kernel (corresponds to activation function type). |
| `_weightVariance` | The weight variance parameter. |

