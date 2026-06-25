---
title: "SpiralConvLayer<T>"
description: "Implements spiral convolution for mesh vertex processing."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements spiral convolution for mesh vertex processing.

## For Beginners

Unlike image convolutions where neighbors are in a grid,
mesh vertices have irregular connectivity. Spiral convolution solves this by:

1. Starting at each vertex
2. Visiting neighbors in a consistent spiral pattern (like a clock hand)
3. Gathering features from each neighbor in order
4. Applying learned weights to the ordered features

This creates a consistent "template" for convolution regardless of mesh topology.

Applications:

- 3D shape analysis and classification
- Facial expression recognition
- Body pose estimation
- Medical surface analysis

## How It Works

SpiralConvLayer operates on mesh vertices by aggregating features from neighbors
in a consistent spiral ordering. This enables translation-equivariant convolutions
on irregular mesh structures by defining a canonical ordering of vertex neighbors.

Reference: "Neural 3D Morphable Models: Spiral Convolutional Networks" by Bouritsas et al.
Reference: "SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator" by Gong et al.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpiralConvLayer(Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `SpiralConvLayer` class. |
| `SpiralConvLayer(Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `SpiralConvLayer` class with vector activation. |
| `SpiralConvLayer(Int32,Int32,Int32,IActivationFunction<>)` | Eager constructor: allocates and initializes the weight/bias tensors immediately for a known input channel count. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputChannels` | Gets the number of input feature channels per vertex. |
| `OutputChannels` | Gets the number of output feature channels per vertex. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SpiralLength` | Gets the spiral sequence length (number of neighbors in the spiral). |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training (backpropagation). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBiases(Tensor<>,Int32)` | Adds biases to each vertex output using vectorized broadcast. |
| `Clone` | Creates a deep copy of this layer. |
| `CombineGatheredFeatures(Tensor<>[],Int32,Int32)` | Combines gathered features from all batch samples for backward pass. |
| `Deserialize(BinaryReader)` | Deserializes the layer from a binary stream. |
| `Dispose(Boolean)` |  |
| `ExtractBatchSlice(Tensor<>,Int32,Int32)` | Extracts a single sample from a batched tensor. |
| `Forward(Tensor<>)` | Performs the forward pass of spiral convolution. |
| `ForwardGpu(Tensor<>[])` |  |
| `GatherSpiralFeatures(Tensor<>,Int32)` | Gathers features from neighbors according to spiral indices. |
| `GetBiases` | Gets the bias tensor. |
| `GetMetadata` | Emits the construction parameters the network's flat-parameter serialization path (GetMetadata + GetParameters, used by Clone / DeepCopy) needs to rebuild this layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWeights` | Gets the weight tensor. |
| `InitializeWeights` | Initializes weights using He (Kaiming) initialization. |
| `OnFirstForward(Tensor<>)` | Resolves input channels and vertex count from input.Shape on first forward (rank-2 [V,C] or rank-3 [B,V,C]) and allocates weights+biases. |
| `ProcessBatched(Tensor<>,Int32,Int32)` | Processes a batched input tensor. |
| `ProcessSingle(Tensor<>,Int32)` | Processes a single (non-batched) input tensor. |
| `ResetState` | Resets cached state from forward/backward passes. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `ScatterSpiralGradients(Tensor<>,Int32)` | Scatters gradients back to input vertices according to spiral indices using vectorized operations. |
| `Serialize(BinaryWriter)` | Serializes the layer to a binary stream. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a vector. |
| `SetSpiralIndices(Int32[0:,0:])` | Sets the spiral indices for the mesh being processed. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates layer parameters using computed gradients. |
| `ValidateParameters(Int32,Int32,Int32)` | Validates constructor parameters. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | Learnable bias values [OutputChannels]. |
| `_biasesGradient` | Cached bias gradients from backward pass. |
| `_gatheredFeatures` | Cached gathered neighbor features for backward pass. |
| `_lastInput` | Cached input from the last forward pass. |
| `_lastOutput` | Cached output from the last forward pass. |
| `_lastPreActivation` | Cached pre-activation output from the last forward pass. |
| `_spiralIndices` | Spiral indices for each vertex [numVertices, SpiralLength]. |
| `_spiralIndicesGpu` | Cached GPU buffer for spiral indices. |
| `_weights` | Learnable weights [OutputChannels, InputChannels * SpiralLength]. |
| `_weightsGradient` | Cached weight gradients from backward pass. |

