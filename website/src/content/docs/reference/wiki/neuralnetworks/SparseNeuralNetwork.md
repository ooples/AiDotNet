---
title: "SparseNeuralNetwork<T>"
description: "Represents a Sparse Neural Network with efficient sparse weight matrices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Sparse Neural Network with efficient sparse weight matrices.

## For Beginners

In a regular neural network, every neuron in one layer is connected
to every neuron in the next layer. In a sparse network, many of these connections are
removed (set to zero), keeping only the most important ones. This has several benefits:

- Uses less memory (only stores non-zero values)
- Runs faster (skips multiplications with zero)
- Can prevent overfitting (acts as regularization)
- Enables very large networks to fit in limited memory

Common use cases include:

- Network compression for mobile/edge deployment
- Recommender systems with sparse user-item matrices
- Graph neural networks with sparse adjacency matrices
- Pruned networks from neural architecture search

## How It Works

A Sparse Neural Network uses sparse weight matrices where most values are zero.
This provides significant memory and computational savings for large networks,
especially when combined with network pruning techniques.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SparseNeuralNetwork` | Initializes a new instance of the SparseNeuralNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the SparseNeuralNetwork with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes sparse neural network-specific data from a binary reader. |
| `Forward(Tensor<>)` | Performs a forward pass through the network with the given input tensor. |
| `GetModelMetadata` | Retrieves metadata about the sparse neural network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the sparse neural network based on the provided architecture. |
| `IsValidInputLayer(ILayer<>)` | Determines if a layer can serve as a valid input layer for this network. |
| `IsValidOutputLayer(ILayer<>)` | Determines if a layer can serve as a valid output layer for this network. |
| `PredictCore(Tensor<>)` | Makes a prediction using the sparse neural network for the given input tensor. |
| `ResolveLearningRate` | Pulls the learning rate from the configured optimizer when available, falling back to 1e-2 (the previous hard-coded value) if no optimizer was supplied at construction time. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes sparse neural network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the sparse neural network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |
| `_sparsity` | The sparsity level (fraction of weights that are zero). |

