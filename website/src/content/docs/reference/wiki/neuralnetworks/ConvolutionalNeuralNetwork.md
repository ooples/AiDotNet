---
title: "ConvolutionalNeuralNetwork<T>"
description: "Represents a Convolutional Neural Network (CNN) that processes multi-dimensional data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Convolutional Neural Network (CNN) that processes multi-dimensional data.

## For Beginners

Think of a CNN as an image recognition system that works similarly to how 
your eyes and brain process visual information. Just as your brain automatically notices 
patterns like edges, shapes, and textures without conscious effort, a CNN automatically 
learns to detect these features in images. This makes CNNs excellent for tasks like 
recognizing objects in photos, detecting faces, or reading handwritten text.

## How It Works

A Convolutional Neural Network is specialized for processing data with a grid-like structure,
such as images. It uses convolutional layers to automatically detect important features
without manual feature extraction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConvolutionalNeuralNetwork` | Initializes a new instance with default architecture settings. |
| `ConvolutionalNeuralNetwork(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,ConvolutionalNeuralNetworkOptions)` | Initializes a new instance of the ConvolutionalNeuralNetwork class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputGradient(Tensor<>,Tensor<>)` | Calculates the gradient of the loss with respect to the network's output. |
| `CreateNewInstance` | Creates a new instance of the convolutional neural network model. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes convolutional neural network-specific data from a binary reader. |
| `Forward(Tensor<>)` | Performs a forward pass through the network with the given input tensor. |
| `GetModelMetadata` | Retrieves metadata about the convolutional neural network model. |
| `GetOptions` |  |
| `GetParameterCount` |  |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Fused conv-stem inference fast path: a canonical CNN classifier — `[Conv(→ReLU) \| MaxPool]+ → Flatten → Dense(+)` — replayed by calling the engine kernels directly (FusedConv2D fusing bias+activation; index-free MaxPool2D; cached-B FusedLin… |
| `PredictEager(Tensor<>)` | Makes a prediction using the convolutional neural network for the given input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes convolutional neural network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the convolutional neural network using the provided input and expected output. |
| `UpdateParameters(List<Tensor<>>)` | Updates the parameters of the network based on the calculated gradients. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |

