---
title: "ExtremeLearningMachine<T>"
description: "Represents an Extreme Learning Machine (ELM), a type of feedforward neural network with a unique training approach."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents an Extreme Learning Machine (ELM), a type of feedforward neural network with a unique training approach.

## For Beginners

An Extreme Learning Machine is like a neural network on fast-forward.

Think of it like building a bridge:

- Traditional neural networks carefully adjust every piece of the bridge (slow but thorough)
- ELMs randomly set up most of the bridge, then only carefully adjust the final section
- This approach is much faster but can still create a surprisingly strong bridge

The "extreme" part refers to its extremely fast training time. While traditional networks
might take hours or days to train, ELMs can often be trained in seconds or minutes, making
them useful for applications where training speed is critical.

## How It Works

An Extreme Learning Machine is a special type of single-hidden-layer feedforward neural network that uses a
non-iterative training approach. Unlike traditional neural networks that use backpropagation to adjust all weights,
ELMs randomly assign the weights between the input and hidden layer and only train the weights between the hidden
and output layer. This is done analytically using a pseudo-inverse operation rather than through iterative
optimization, resulting in extremely fast training times while maintaining good generalization performance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExtremeLearningMachine` | Initializes a new instance of the `ExtremeLearningMachine` class with the specified architecture and hidden layer size. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePseudoInverse(Matrix<>)` | Calculates the Moore-Penrose pseudoinverse of a matrix. |
| `CreateNewInstance` | Creates a new instance of the ExtremeLearningMachine with the same configuration as the current instance. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Extreme Learning Machine-specific data from a binary reader. |
| `GetModelMetadata` | Gets metadata about the Extreme Learning Machine model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the Extreme Learning Machine based on the architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the Extreme Learning Machine. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Extreme Learning Machine-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the Extreme Learning Machine on a single batch of data. |
| `TrainWithRegularization(Tensor<>,Tensor<>,Double)` | Trains the ELM using regularized least squares for improved generalization. |
| `UpdateOutputLayerWeights(Matrix<>)` | Updates the weights of the output layer with the calculated weights. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the Extreme Learning Machine. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_hiddenLayerSize` | Gets the size of the hidden layer (number of neurons). |

