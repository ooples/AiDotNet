---
title: "MeasurementLayer<T>"
description: "Represents a layer that performs quantum measurement operations on complex-valued input tensors."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that performs quantum measurement operations on complex-valued input tensors.

## For Beginners

This layer converts quantum information into regular probabilities.

Think of it like a bridge between the quantum and classical worlds:

- In quantum computing, information exists in "superposition" (multiple states at once)
- This layer converts that quantum information into classical probabilities
- It's similar to how quantum physics says we can only observe probabilities in the real world

For example, if you have a quantum state representing a coin that's in both heads and tails
at the same time, the measurement layer would convert this to classical probabilities like
"60% chance of heads, 40% chance of tails."

This is a fundamental concept in quantum computing and quantum mechanics.

## How It Works

The MeasurementLayer transforms complex-valued quantum state amplitudes into classical probabilities.
It calculates the probability distribution from a quantum state vector by taking the squared magnitude
of each complex amplitude and normalizing the results to ensure they sum to 1.0.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeasurementLayer(Int32)` | Initializes a new instance of the `MeasurementLayer` class with the specified size. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the measurement layer. |
| `ForwardGpu(Tensor<>[])` | Performs the GPU-accelerated forward pass for quantum measurement. |
| `GetParameters` | Gets all trainable parameters from the measurement layer as a single vector. |
| `ResetState` | Resets the internal state of the measurement layer. |
| `UpdateParameters()` | Updates the parameters of the measurement layer using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastInput` | The input tensor from the most recent forward pass. |
| `_lastOutput` | The output tensor from the most recent forward pass. |

