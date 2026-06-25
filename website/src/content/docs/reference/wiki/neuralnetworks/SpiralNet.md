---
title: "SpiralNet<T>"
description: "Implements the SpiralNet++ architecture for mesh-based deep learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements the SpiralNet++ architecture for mesh-based deep learning.

## For Beginners

SpiralNet++ is designed for learning from 3D mesh data.

Key concepts:

- Mesh: A 3D surface made of vertices connected by edges/triangles
- Spiral ordering: A consistent way to visit vertex neighbors (like a clock hand)
- Spiral convolution: Apply weights to neighbors in spiral order

How it works:

1. For each vertex, define a spiral ordering of its neighbors
2. Gather neighbor features in spiral order
3. Apply learned weights to the gathered features
4. Pool vertices to create hierarchical representations
5. Classify or segment the mesh

Applications:

- 3D face reconstruction and expression recognition
- Human body shape analysis
- Medical surface analysis (organs, bones)
- CAD model classification

## How It Works

SpiralNet++ processes 3D meshes by applying convolutions along spiral sequences
of vertex neighbors. This creates translation-equivariant operations on
irregular mesh structures without requiring mesh registration.

Reference: "SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator" by Gong et al.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpiralNet` | Initializes a new instance of the `SpiralNet` class with default options. |
| `SpiralNet(Int32,Int32,Int32,ILossFunction<>)` | Initializes a new instance of the `SpiralNet` class with simple parameters. |
| `SpiralNet(SpiralNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the `SpiralNet` class with specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvChannels` | Gets the channel configuration for spiral convolution layers. |
| `InputFeatures` | Gets the number of input features per vertex. |
| `NumClasses` | Gets the number of output classes for classification. |
| `SpiralLength` | Gets the spiral sequence length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateArchitecture(SpiralNetOptions)` | Creates the neural network architecture for SpiralNet. |
| `CreateNewInstance` | Creates a new instance for cloning. |
| `CreateOneHotTarget(Int32,Int32)` | Creates a one-hot encoded target vector. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` | Gets metadata about this model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the SpiralNet network. |
| `PredictClass(Tensor<>,Int32[0:,0:])` | Predicts the class for a single mesh. |
| `PredictCore(Tensor<>)` | Generates predictions for the given input. |
| `PredictProbabilities(Tensor<>,Int32[0:,0:])` | Computes class probabilities for a single mesh using softmax. |
| `PropagateSpiralIndicesToLayers` | Propagates spiral indices to all SpiralConv layers. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `SetMultiResolutionSpiralIndices(List<Int32[0:,0:]>)` | Sets spiral indices for multiple resolution levels (for hierarchical processing). |
| `SetSpiralIndices(Int32[0:,0:])` | Sets the spiral indices for the current mesh being processed. |
| `Train(List<Tensor<>>,List<Int32[0:,0:]>,List<Int32>,Int32,)` | Trains the network on mesh data. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch. |
| `UpdateParameters()` | Updates network parameters using the optimizer. |
| `UpdateParameters(Vector<>)` | Updates network parameters using a flat parameter vector. |
| `ValidateCustomLayers(IList<ILayer<>>)` | Validates custom layers for compatibility with SpiralNet architecture. |
| `ValidateOptions(SpiralNetOptions)` | Validates configuration options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lossFunction` | The loss function used to compute training loss. |
| `_optimizer` | The optimizer used to update network parameters. |
| `_options` | The configuration options for this SpiralNet instance. |
| `_spiralIndicesPerLevel` | Cached spiral indices for each resolution level. |

