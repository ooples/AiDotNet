---
title: "MeshEdgeConvLayer<T>"
description: "Implements edge convolution for mesh-based neural networks (MeshCNN style)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements edge convolution for mesh-based neural networks (MeshCNN style).

## For Beginners

Think of a mesh as a surface made of connected triangles.
Each triangle edge is shared by (at most) two triangles. This layer examines each edge
and the two triangles it connects to learn meaningful features about the shape.

Key concepts:

- Edge: A line segment connecting two vertices, shared by up to 2 faces
- Dihedral angle: The angle between two faces sharing an edge
- Edge features: Properties like length, angles, and face normals

The layer learns to recognize patterns in how faces connect, enabling recognition
of shapes, surface curvature, and other geometric properties.

## How It Works

MeshEdgeConvLayer processes triangle meshes by treating edges as the fundamental unit.
Each edge connects two triangular faces, and the layer learns to extract features
from the geometric relationship between these faces.

Reference: "MeshCNN: A Network with an Edge" by Hanocka et al., SIGGRAPH 2019

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeshEdgeConvLayer(Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `MeshEdgeConvLayer` class. |
| `MeshEdgeConvLayer(Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `MeshEdgeConvLayer` class with vector activation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputChannels` | Gets the number of input feature channels per edge. |
| `NumNeighbors` | Gets the number of neighboring edges to consider for each edge convolution. |
| `OutputChannels` | Gets the number of output feature channels per edge. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training (backpropagation). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBiases(Tensor<>)` | Adds bias values to each output channel using vectorized operations. |
| `AggregateEdgeFeatures(Tensor<>,Int32[0:,0:],Int32,Int32)` | Aggregates features from each edge and its neighbors using vectorized operations. |
| `Clone` | Creates a deep copy of the layer. |
| `ComputeBiasGradient(Tensor<>)` | Computes the bias gradient by summing gradients over edges using vectorized reduction. |
| `Deserialize(BinaryReader)` | Deserializes the layer from a binary stream. |
| `Forward(Tensor<>)` | Performs the forward pass of edge convolution. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass for edge convolution. |
| `GetBiases` | Gets the bias tensor. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWeights` | Gets the weight tensor. |
| `InitializeWeights` | Initializes weights using He (Kaiming) uniform initialization and biases to zero. |
| `ResetState` | Resets the cached state from forward/backward passes. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `ScatterGradients(Tensor<>,Int32[0:,0:],Int32)` | Scatters gradients from aggregated features back to original edges using vectorized operations. |
| `Serialize(BinaryWriter)` | Serializes the layer to a binary stream. |
| `SetEdgeAdjacency(Int32[0:,0:])` | Sets the edge adjacency information for the current mesh. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer parameters using the computed gradients and learning rate. |
| `ValidateParameters(Int32,Int32,Int32)` | Validates constructor parameters. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | Learnable bias values [OutputChannels]. |
| `_biasesGradient` | Cached bias gradients from backward pass. |
| `_lastEdgeAdjacency` | Cached edge adjacency from the last forward pass. |
| `_lastInput` | Cached input from the last forward pass. |
| `_lastOutput` | Cached output after activation from the last forward pass. |
| `_lastPreActivation` | Cached output before activation from the last forward pass. |
| `_weights` | Learnable weights for edge convolution [OutputChannels, InputChannels * (1 + NumNeighbors)]. |
| `_weightsGradient` | Cached weight gradients from backward pass. |

