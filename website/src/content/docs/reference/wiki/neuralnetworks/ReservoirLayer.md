---
title: "ReservoirLayer<T>"
description: "Represents a reservoir layer used in Echo State Networks (ESNs) for processing sequential data with fixed random weights."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a reservoir layer used in Echo State Networks (ESNs) for processing sequential data with fixed random weights.

## For Beginners

This layer works like a complex echo chamber for your data.

Think of the ReservoirLayer as a special room that creates rich echoes:

- When you speak a word into this room (input data), it creates complex echoes (reservoir state)
- These echoes depend both on what you just said and on the echoes of previous words
- The room's shape and materials (reservoir weights) determine how echoes form and persist
- Unlike other neural networks, the room's properties are fixed and don't change during training

For example, when processing a sentence word by word:

- Each word causes a unique pattern of echoes in the reservoir
- These echoes contain information about both the current word and previous words
- The patterns are rich enough that a simple output layer can be trained to extract useful information

This approach is powerful because:

- The random, fixed reservoir creates complex transformations of the input data
- Only the output layer needs to be trained, making learning faster and simpler
- It works especially well for time series prediction and certain sequence processing tasks

Echo State Networks are particularly effective when you need to model complex dynamical systems
with a simpler training process than traditional recurrent neural networks.

## How It Works

The ReservoirLayer implements the core component of an Echo State Network, a type of recurrent neural network
where the internal connections (reservoir weights) are randomly initialized and remain fixed during training.
This layer maintains a high-dimensional reservoir state that is updated based on the current input and the
previous state. The key characteristic of an ESN is that only the output layer is trained, while the reservoir
itself remains unchanged.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReservoirLayer(Int32,Int32,Double,Double,Double,Double,IInitializationStrategy<>)` | Initializes a new instance of the `ReservoirLayer` class with specified dimensions and properties. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Updates the parameters of the reservoir layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDotProduct(Tensor<>,Tensor<>)` | Computes the dot product of two tensors. |
| `ComputeMaxEigenvalue(Tensor<>)` | Computes the maximum eigenvalue (spectral radius) of a matrix using power iteration. |
| `ComputeNorm(Tensor<>)` | Computes the L2 norm of a tensor. |
| `Forward(Tensor<>)` | Performs the forward pass of the reservoir layer. |
| `ForwardGpu(Tensor<>[])` | Performs the GPU-accelerated forward pass for the reservoir layer. |
| `GetParameters` | Gets all parameters of the reservoir layer as a single vector. |
| `GetState` | Gets the current state of the reservoir. |
| `InitializeReservoir` | Initializes the reservoir weights and state with proper scaling. |
| `ResetState` | Resets the internal state of the reservoir layer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_connectionProbability` | The probability of connection between any two neurons in the reservoir. |
| `_inputScaling` | The scaling factor applied to input before it enters the reservoir. |
| `_inputSize` | The size of the input vector at each time step. |
| `_inputWeights` | The input weight tensor mapping inputs into the reservoir space. |
| `_leakingRate` | The leaking rate determining how quickly the reservoir state updates. |
| `_reservoirSize` | The size of the reservoir, determining the dimensionality of the reservoir state. |
| `_reservoirState` | The current state of the reservoir, representing the activation of all neurons. |
| `_reservoirWeights` | The weight tensor representing connections between neurons in the reservoir. |
| `_spectralRadius` | The spectral radius of the reservoir weight matrix, affecting the memory of the network. |

