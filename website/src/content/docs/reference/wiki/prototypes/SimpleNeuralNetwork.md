---
title: "SimpleNeuralNetwork<T>"
description: "Simple 2-layer neural network for prototype validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Prototypes`

Simple 2-layer neural network for prototype validation.
Demonstrates GPU acceleration through vectorized operations.

## How It Works

This is a minimal neural network for Phase A prototype validation.
It includes:

- Input layer → Hidden layer (with ReLU activation)
- Hidden layer → Output layer (linear)
- Backpropagation
- Adam optimizer integration

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimpleNeuralNetwork(Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the SimpleNeuralNetwork. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of parameters in the network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyReLU(PrototypeVector<>)` | Applies ReLU activation: max(0, x) |
| `ApplyReLUDerivative(PrototypeVector<>,PrototypeVector<>)` | Applies ReLU derivative: gradient * (x > 0) |
| `Backward(PrototypeVector<>)` | Backward pass (computes gradients). |
| `ComputeLoss(PrototypeVector<>,PrototypeVector<>)` | Computes mean squared error loss. |
| `ComputeLossGradient(PrototypeVector<>,PrototypeVector<>)` | Computes gradient of MSE loss. |
| `CreateRandomVector(Int32,Double)` | Creates a random vector with values from normal distribution. |
| `Forward(PrototypeVector<>)` | Forward pass through the network. |
| `GetParameters` | Gets all parameters as a single flattened vector. |
| `InitializeWeights` | Initializes weights using Xavier/Glorot initialization. |
| `MatrixVectorMultiply(PrototypeVector<>,PrototypeVector<>,Int32,Int32)` | Matrix-vector multiplication: result = matrix @ vector |
| `MatrixVectorMultiplyTranspose(PrototypeVector<>,PrototypeVector<>,Int32,Int32)` | Matrix-vector multiplication with transposed matrix: result = matrix^T @ vector Matrix is (rows, cols), transpose is (cols, rows), result is (cols) |
| `OuterProduct(PrototypeVector<>,PrototypeVector<>)` | Outer product: result = a @ b^T (flattened) |
| `SetParameters(PrototypeVector<>)` | Sets all parameters from a single flattened vector. |

