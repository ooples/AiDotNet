---
title: "GraphNeuralOperator<T>"
description: "Implements Graph Neural Operators for learning operators on graph-structured data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.NeuralOperators`

Implements Graph Neural Operators for learning operators on graph-structured data.

## How It Works

For Beginners:
Graph Neural Operators extend neural operators to irregular, graph-structured domains.

Why Graphs?
Many physical systems are naturally represented as graphs:

- Molecular structures (atoms = nodes, bonds = edges)
- Mesh-based simulations (mesh points = nodes, connectivity = edges)
- Traffic networks (intersections = nodes, roads = edges)
- Social networks, power grids, etc.

Regular operators (FNO, DeepONet) work on:

- Structured grids (images, regular spatial domains)
- Euclidean spaces

Graph operators work on:

- Irregular geometries
- Non-Euclidean spaces
- Variable-size domains

Key Idea - Message Passing:
Information propagates through the graph via message passing:

1. Each node has features (e.g., temperature, velocity)
2. Nodes send messages to neighbors
3. Nodes aggregate messages and update their features
4. Repeat for multiple layers

Applications:

- Molecular dynamics (predict molecular properties)
- Computational fluid dynamics (irregular meshes)
- Material science (crystal structures)
- Climate modeling (irregular Earth grids)
- Particle systems

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphNeuralOperator` | Initializes a new instance with default architecture settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of parameters across graph layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes graph operator-specific data. |
| `Forward(Tensor<>)` | Forward pass using an identity adjacency matrix. |
| `Forward([0:,0:],[0:,0:])` | Forward pass through the graph neural operator. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetModelMetadata` | Gets metadata about the graph neural operator. |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `GetParameters` | Gets the operator parameters as a flattened vector. |
| `PredictCore(Tensor<>)` | Makes a prediction using the graph neural operator. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes graph operator-specific data. |
| `Train(Tensor<>,Tensor<>)` | Performs a basic supervised training step using MSE loss. |
| `UpdateParameters(Vector<>)` | Updates the operator parameters from a flattened vector. |

