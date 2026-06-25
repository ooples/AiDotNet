---
title: "ResidualNeuralNetwork<T>"
description: "Represents a Residual Neural Network, which is a type of neural network that uses skip connections to address the vanishing gradient problem in deep networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Residual Neural Network, which is a type of neural network that uses skip connections to address the vanishing gradient problem in deep networks.

## For Beginners

A Residual Neural Network is like a highway system for information in a neural network.

Think of it like this:

- In a traditional neural network, information must pass through every layer sequentially
- In a ResNet, there are "shortcut paths" or "highways" that let information skip ahead

For example, imagine trying to pass a message through a line of 100 people:

- In a regular network, each person must whisper to the next person in line
- In a ResNet, some people can also shout directly to someone 5 positions ahead

This design solves a major problem: in very deep networks (many layers), information and learning signals
tend to fade away or "vanish" as they travel through many layers. The shortcuts in ResNets help information
flow more easily through the network, allowing for much deeper networks (some with over 100 layers!)
that can learn more complex patterns.

ResNets revolutionized image recognition and are now used in many AI systems that need to identify
complex patterns in data.

## How It Works

A Residual Neural Network (ResNet) is an advanced neural network architecture that introduces "skip connections" or "shortcuts"
that allow information to bypass one or more layers. These residual connections help address the vanishing gradient problem
that occurs in very deep networks, enabling the training of networks with many more layers than previously possible.
ResNets were a breakthrough in deep learning that significantly improved performance on image recognition and other tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResidualNeuralNetwork` | Initializes a new instance of the `ResidualNeuralNetwork` class with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the deep supervision auxiliary loss. |
| `SupportsTraining` | Indicates whether this network supports training (learning from data). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAuxiliaryClassifier(ILayer<>,Int32)` | Adds an auxiliary classifier at the specified layer position for deep supervision. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for deep supervision from intermediate auxiliary classifiers. |
| `CreateNewInstance` | Creates a new instance of the residual neural network with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data for the Residual Neural Network. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the deep supervision auxiliary loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Gets metadata about the Residual Neural Network model. |
| `GetOptions` |  |
| `InitializeAuxiliaryClassifiers` | Automatically initializes auxiliary classifiers at strategic positions based on network depth. |
| `InitializeLayers` | Initializes the neural network layers based on the provided architecture or default configuration. |
| `PredictCore(Tensor<>)` | Makes a prediction using the Residual Neural Network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data for the Residual Neural Network. |
| `Train(Tensor<>,Tensor<>)` | Trains the Residual Neural Network on the provided data. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the residual neural network layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_batchSize` | Gets or sets the batch size for training. |
| `_epochs` | Gets or sets the number of training epochs. |
| `_learningRate` | Gets or sets the learning rate for parameter updates. |
| `_trainOptimizer` | Persistent Adam optimizer for Train() calls. |

