---
title: "QuantumLayer<T>"
description: "Represents a neural network layer that uses quantum computing principles for processing inputs."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a neural network layer that uses quantum computing principles for processing inputs.

## For Beginners

This layer uses concepts from quantum computing to process data in a unique way.

Think of it like a special filter that:

- Transforms regular data into a quantum-like format (similar to how light can be both a wave and a particle)
- Performs calculations that explore multiple possibilities simultaneously
- Converts the results back into standard values that other layers can work with

While traditional neural networks work with definite values, quantum layers work with probabilities
and superpositions (being in multiple states at once). This can help the network find patterns
that might be missed with traditional approaches.

You don't need to understand quantum physics to use this layer - just know that it offers a
different way of processing information that can be powerful for certain problems.

## How It Works

The QuantumLayer implements a simulated quantum circuit that processes input data using quantum
rotations and measurements. It transforms classical inputs into quantum states, applies quantum
operations, and converts the results back to classical outputs. This approach can potentially
capture complex patterns that traditional neural network layers might miss.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuantumLayer(Int32,Int32,Int32)` | Initializes a new instance of the `QuantumLayer` class with specified dimensions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyRotation(Int32,)` | Applies a rotation operation to a specific qubit in the quantum circuit. |
| `Forward(Tensor<>)` | Performs the forward pass of the quantum layer. |
| `ForwardGpu(Tensor<>[])` | Performs the GPU-accelerated forward pass for the quantum layer. |
| `GetMetadata` | Updates the parameters of the quantum layer using the calculated gradients. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the quantum layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeQuantumCircuit` | Initializes the quantum circuit with an identity matrix and random rotation angles. |
| `ResetQuantumCircuit` | Resets the quantum circuit to an identity matrix. |
| `ResetState` | Resets the internal state of the quantum layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the quantum layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateAngleGradients(Tensor<Complex<>>,Int32)` | Updates the gradients of the rotation angles based on the output gradient. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastResultReal` | Cached result amplitudes from Forward for use in Backward. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

