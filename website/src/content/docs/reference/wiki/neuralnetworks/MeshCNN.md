---
title: "MeshCNN<T>"
description: "Implements the MeshCNN architecture for processing 3D triangle meshes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements the MeshCNN architecture for processing 3D triangle meshes.

## For Beginners

MeshCNN processes 3D shapes represented as triangle meshes.

Key concepts:

- Mesh: A 3D surface made of connected triangles (vertices + faces)
- Edge: A line segment connecting two vertices, shared by up to 2 faces
- Edge features: Properties like dihedral angle, edge ratios, face angles

How it works:

1. Extract edge features from the mesh (5 features per edge by default)
2. Apply edge convolutions to learn patterns in edge neighborhoods
3. Pool edges by removing less important ones (simplifies the mesh)
4. Repeat conv + pool to build hierarchical features
5. Aggregate edge features for classification/segmentation

Applications:

- 3D shape classification (e.g., recognize chair vs table)
- Mesh segmentation (label each part of a 3D model)
- Shape retrieval (find similar 3D models)

## How It Works

MeshCNN is a deep learning architecture that operates directly on 3D mesh data
by treating edges as the fundamental unit of computation. This enables learning
from the mesh structure itself rather than converting to voxels or point clouds.

Reference: "MeshCNN: A Network with an Edge" by Hanocka et al., SIGGRAPH 2019

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeshCNN` | Initializes a new instance of the `MeshCNN` class with default options. |
| `MeshCNN(Int32,Int32,ILossFunction<>)` | Initializes a new instance of the `MeshCNN` class with simple parameters. |
| `MeshCNN(MeshCNNOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the `MeshCNN` class with specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvChannels` | Gets the channel configuration for edge convolution layers. |
| `InputFeatures` | Gets the number of input features per edge. |
| `NumClasses` | Gets the number of output classes for classification. |
| `PoolTargets` | Gets the pooling targets for mesh simplification. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateArchitecture(MeshCNNOptions)` | Creates the neural network architecture for MeshCNN. |
| `CreateNewInstance` | Creates a new instance for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetModelMetadata` | Gets metadata about this model. |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the MeshCNN network. |
| `PredictCore(Tensor<>)` | Generates predictions for the given input. |
| `PropagateAdjacencyToLayers` | Propagates edge adjacency to all MeshEdgeConv and MeshPool layers. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `SetEdgeAdjacency(Int32[0:,0:])` | Sets the edge adjacency for the current mesh being processed. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch. |
| `UpdateParameters(Vector<>)` | Updates network parameters using a flat parameter vector. |
| `UpdateSubsequentLayerAdjacency(ILayer<>,Int32[0:,0:])` | Updates adjacency for layers following a pooling operation. |
| `ValidateOptions(MeshCNNOptions)` | Validates configuration options. |
| `WalkLayersWithMeshReshape(Tensor<>,Dictionary<String,Tensor<>>)` | Walks `Layers` applying the MeshCNN-specific rank promotion required by `GlobalPoolingLayer`. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_currentEdgeAdjacency` | Cached edge adjacency for the current mesh being processed. |
| `_lossFunction` | The loss function used to compute training loss. |
| `_optimizer` | The optimizer used to update network parameters. |
| `_options` | The configuration options for this MeshCNN instance. |

