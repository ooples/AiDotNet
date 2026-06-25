---
title: "GraphNeuralNetwork<T>"
description: "Represents a Graph Neural Network that can process data represented as graphs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Graph Neural Network that can process data represented as graphs.

## For Beginners

A Graph Neural Network is a type of neural network that works with connected data.

Think of it like analyzing a social network:

- Each person is a "node" in the graph
- Friendships between people are "edges" connecting the nodes
- People have attributes (like age, location, interests) which are "node features"

GNNs are useful when:

- The relationships between items are as important as the items themselves
- You're working with network-like data (social networks, molecules, road systems)
- You need to make predictions about how nodes influence each other

For example, GNNs can help predict which products a customer might like based on
what similar customers have purchased, by analyzing the connections between customers and products.

## How It Works

A Graph Neural Network (GNN) is designed to work with data structured as graphs, 
where nodes represent entities and edges represent relationships between these entities.
This implementation supports various activation functions for different layers and
provides methods for predicting outputs from both vector inputs and graph inputs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphNeuralNetwork` | Initializes a new instance of the `GraphNeuralNetwork` class with vector activation functions. |
| `GraphNeuralNetwork(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,IActivationFunction<>,IActivationFunction<>,IActivationFunction<>,IActivationFunction<>,GraphNeuralNetworkOptions)` | Initializes a new instance of the `GraphNeuralNetwork` class with scalar activation functions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the graph smoothness auxiliary loss. |
| `UseAuxiliaryLoss` | Gets or sets whether auxiliary loss (graph smoothness regularization) should be used during training. |
| `_activationLayerScalarActivation` | Gets or sets the scalar activation function used in standard activation layers. |
| `_activationLayerVectorActivation` | Gets or sets the vector activation function used in standard activation layers. |
| `_finalActivationLayerScalarActivation` | Gets or sets the scalar activation function used in the final activation layer. |
| `_finalActivationLayerVectorActivation` | Gets or sets the vector activation function used in the final activation layer. |
| `_finalDenseLayerScalarActivation` | Gets or sets the scalar activation function used in the final dense layer. |
| `_finalDenseLayerVectorActivation` | Gets or sets the vector activation function used in the final dense layer. |
| `_graphConvolutionalScalarActivation` | Gets or sets the scalar activation function used in graph convolutional layers. |
| `_graphConvolutionalVectorActivation` | Gets or sets the vector activation function used in graph convolutional layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for graph smoothness regularization. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Graph Neural Network-specific data from a binary reader. |
| `GetActivationTypes` | Gets the types of activation functions used in the network. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the graph smoothness auxiliary loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetNamedLayerActivations(Tensor<>)` | Gets metadata about the Graph Neural Network model. |
| `GetOptions` |  |
| `HybridPooling(Tensor<>)` | Performs hybrid pooling on node features to generate a final output representation. |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Performs a forward pass through the network to make a prediction using a standard input tensor. |
| `PredictGraph(Tensor<>,Tensor<>)` | Performs a forward pass through the network to generate a prediction from graph data. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Graph Neural Network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the Graph Neural Network on a single input-output pair. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network using the provided parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultTrainLearningRate` | Default Adam learning rate for GNN training. |
| `_autoAdjacencyMatrix` | Cached adjacency matrix for forward/backward passes. |

