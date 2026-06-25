---
title: "FeedForwardNeuralNetwork<T>"
description: "Represents a Feed-Forward Neural Network (FFNN) for processing data in a forward path."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Feed-Forward Neural Network (FFNN) for processing data in a forward path.

## For Beginners

Think of a Feed-Forward Neural Network like a series of information-processing
stages arranged in a line. Data flows only forward through these stages, never backward. Each stage
(or layer) processes the information and passes it to the next stage. This simple structure makes
FFNNs great for many common tasks like classification (deciding which category something belongs to)
or regression (predicting a numerical value).

## How It Works

A Feed-Forward Neural Network is the simplest type of artificial neural network, where connections
between nodes do not form a cycle. Information moves in only one direction -- forward -- from the input
nodes, through the hidden nodes (if any), and to the output nodes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeedForwardNeuralNetwork` | Initializes a new instance of the FeedForwardNeuralNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the FeedForwardNeuralNetwork with the same configuration as the current instance. |
| `Forward(Tensor<>)` | Performs a forward pass through the network with the given input tensor. |
| `GetModelMetadata` | Retrieves metadata about the feed-forward neural network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the feed-forward neural network for the given input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes feed-forward neural network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the feed-forward neural network using the provided input and expected output. |
| `TryCompiledMlpPredict(Tensor<>,List<Tensor<>>,List<Tensor<>>,FusedActivationType,FusedActivationType,Tensor<>)` | Runs the pure-dense float stack through the cached `CompiledMlp` plan. |
| `TryFusedDensePredict(Tensor<>,Tensor<>)` | Attempts the fused multi-layer-perceptron inference kernel for a pure dense+activation stack. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |
| `ValidateInputShape(Tensor<>,String)` | Accepts either an unbatched input matching `Architecture.GetInputShape()` or a batched input `[B, ...expectedShape]` whose trailing dims match each axis exactly. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |

