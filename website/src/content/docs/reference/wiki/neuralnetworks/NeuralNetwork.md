---
title: "NeuralNetwork<T>"
description: "A neural network implementation that processes data through multiple layers to make predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

A neural network implementation that processes data through multiple layers to make predictions.

## For Beginners

A neural network is like a brain-inspired system that learns from examples.

Think of a neural network as an assembly line for information:

- Input data enters the "factory" (like features of an image or text)
- It passes through several processing stations (layers of neurons)
- Each station transforms the information in specific ways
- Finally, it produces an output (like a prediction or classification)

For example, if you want to classify images of cats and dogs:

- The input would be the pixel values of the image
- The neural network processes these values through its layers
- Each layer learns to recognize different patterns (edges, shapes, textures, etc.)
- The output tells you the probability of the image containing a cat or dog

The network "learns" by adjusting its internal parameters based on examples,
gradually improving its predictions through a process called training.

## How It Works

Neural networks are computing systems inspired by the human brain. They consist of multiple layers
of interconnected nodes (neurons) that process input data to produce predictions. This class provides
a straightforward implementation that can be used for various machine learning tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralNetwork` | Creates a new neural network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this network supports training (learning from data). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the neural network with the same architecture. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes neural network-specific data from a binary reader. |
| `GetModelMetadata` | Gets metadata about the neural network. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the neural network based on the architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the neural network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes neural network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the neural network on input-output pairs. |
| `UpdateParameters(Vector<>)` | Updates the parameters (weights and biases) of the neural network. |

