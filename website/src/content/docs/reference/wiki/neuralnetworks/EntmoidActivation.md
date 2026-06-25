---
title: "EntmoidActivation<T>"
description: "Entmoid activation function for NODE architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Entmoid activation function for NODE architecture.

## For Beginners

Entmoid is like a "smart sigmoid":

- Regular sigmoid smoothly goes from 0 to 1
- Entmoid can produce exact zeros (sparse output)
- This helps the model focus on important features and ignore noise

The alpha parameter controls sparsity:

- alpha = 1.5 (default): moderately sparse
- alpha = 2: sparsemax (can produce exact zeros)
- alpha → 1: approaches softmax (no sparsity)

## How It Works

Entmoid is a sparse, differentiable activation that generalizes softmax to produce
sparse outputs. It's the element-wise version of entmax (sparse attention).
The function is defined as: entmoid(x) = max(0, (alpha * x + 1) / (2 * alpha))^(1/(alpha-1))

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EntmoidActivation(Double)` | Initializes the entmoid activation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the alpha parameter controlling sparsity. |
| `SupportsGpuTraining` | Gets whether entmoid supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the entmoid activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the entmoid activation function to a tensor. |
| `Activate(Vector<>)` | Applies the entmoid activation function to a vector. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Computes the derivative of the entmoid activation. |
| `Derivative(Tensor<>)` | Computes the derivative of entmoid for a tensor. |
| `Derivative(Vector<>)` | Computes the Jacobian matrix of entmoid for a vector. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the entmoid activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

