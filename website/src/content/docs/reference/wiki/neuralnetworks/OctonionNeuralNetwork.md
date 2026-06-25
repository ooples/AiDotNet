---
title: "OctonionNeuralNetwork<T>"
description: "Represents an Octonion-valued Neural Network for processing data in 8-dimensional hypercomplex space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents an Octonion-valued Neural Network for processing data in 8-dimensional hypercomplex space.

## For Beginners

Octonions are 8-dimensional numbers that extend complex numbers and quaternions.
While regular neural networks use simple numbers, octonion networks use these 8-dimensional numbers
which can capture more complex relationships in data. This is particularly useful for:

- 3D graphics and physics simulations
- Signal processing with multiple channels
- Tasks requiring rich rotational representations

## How It Works

An Octonion Neural Network uses octonion algebra (8-dimensional non-associative division algebra)
for its computations. This provides richer representational capacity than real, complex, or
quaternion-valued networks, making it suitable for tasks requiring high-dimensional rotations
and transformations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OctonionNeuralNetwork` | Initializes a new instance of the OctonionNeuralNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the OctonionNeuralNetwork with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes octonion neural network-specific data from a binary reader. |
| `Forward(Tensor<>)` | Performs a forward pass through the network with the given input tensor. |
| `GetModelMetadata` | Retrieves metadata about the octonion neural network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the octonion neural network based on the provided architecture. |
| `IsValidInputLayer(ILayer<>)` | Determines if a layer can serve as a valid input layer for this network. |
| `IsValidOutputLayer(ILayer<>)` | Determines if a layer can serve as a valid output layer for this network. |
| `PredictCore(Tensor<>)` | Makes a prediction using the octonion neural network for the given input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes octonion neural network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the octonion neural network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |

