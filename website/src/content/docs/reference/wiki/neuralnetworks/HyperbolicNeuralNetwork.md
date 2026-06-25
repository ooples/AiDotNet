---
title: "HyperbolicNeuralNetwork<T>"
description: "Represents a Hyperbolic Neural Network for learning hierarchical representations in Poincare ball space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Hyperbolic Neural Network for learning hierarchical representations in Poincare ball space.

## For Beginners

Hyperbolic neural networks are designed for data that has a natural
hierarchy or tree-like structure. Examples include:

- Taxonomies (e.g., animal kingdom classification)
- Organizational hierarchies
- Social networks with community structures
- Knowledge graphs

In hyperbolic space, the "distance" near the center is smaller than near the edges,
allowing hierarchies to be represented more efficiently than in regular flat space.
Points near the center represent "root" concepts, while points near the edge represent
more specific "leaf" concepts.

## How It Works

A Hyperbolic Neural Network operates in hyperbolic space (specifically the Poincare ball model)
rather than Euclidean space. This allows it to naturally capture hierarchical and tree-like
structures in data with lower distortion than traditional networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperbolicNeuralNetwork` | Initializes a new instance of the HyperbolicNeuralNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the HyperbolicNeuralNetwork with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes hyperbolic neural network-specific data from a binary reader. |
| `Forward(Tensor<>)` | Performs a forward pass through the network with the given input tensor. |
| `GetModelMetadata` | Retrieves metadata about the hyperbolic neural network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the hyperbolic neural network based on the provided architecture. |
| `IsValidInputLayer(ILayer<>)` | Determines if a layer can serve as a valid input layer for this network. |
| `IsValidOutputLayer(ILayer<>)` | Determines if a layer can serve as a valid output layer for this network. |
| `PredictCore(Tensor<>)` | Makes a prediction using the hyperbolic neural network for the given input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes hyperbolic neural network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the hyperbolic neural network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_curvature` | The curvature of the hyperbolic space (must be negative). |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |

