---
title: "HopfieldNetwork<T>"
description: "Represents a Hopfield Network, a recurrent neural network designed for pattern storage and retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Hopfield Network, a recurrent neural network designed for pattern storage and retrieval.

## For Beginners

A Hopfield Network is like a special memory system that can store and recall patterns.

Think of it like a photo album with a magical property:

- You can store a collection of complete photos in the album
- Later, if you show the album a damaged or partial photo, it can recall the complete original version

For example:

- You might store clear images of the digits 0-9
- If you later show the network a smudged or partially erased "7", it can recall the clean version

Hopfield networks work differently from most neural networks:

- They don't have separate input and output layers
- All neurons are connected to each other (but not to themselves)
- They use a special learning rule based on correlations between pattern elements

These networks are useful for tasks like:

- Image reconstruction
- Pattern recognition
- Noise filtering
- Solving certain optimization problems

## How It Works

A Hopfield Network is a type of recurrent artificial neural network that serves as a content-addressable memory system.
It can store patterns and retrieve them based on partial or noisy inputs. The network consists of a single layer
of fully connected neurons, with each neuron connected to all others except itself.
Hopfield networks are particularly useful for pattern recognition, image restoration, and optimization problems.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HopfieldNetwork` | Initializes a new instance of the `HopfieldNetwork` class with the specified architecture and size. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` | Converts an input tensor to a vector, performs pattern recall, and converts back to a tensor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateEnergy(Vector<>)` | Calculates the energy of the current state of the Hopfield network. |
| `CreateNewInstance` | Creates a new instance of the Hopfield Network with the same architecture and configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Hopfield network-specific data from a binary reader. |
| `GetModelMetadata` | Gets metadata about the Hopfield network model. |
| `GetNamedLayerActivations(Tensor<>)` | Returns the weight matrix state as a named activation (Hopfield has no layers). |
| `GetNetworkCapacity` | Gets the maximum number of patterns that can be reliably stored in the network. |
| `GetOptions` |  |
| `GetParameters` |  |
| `InitializeLayers` | Initializes the layers of the neural network. |
| `InitializeWeights` | Initializes the weight matrix with zeros. |
| `Recall(Vector<>,Int32)` | Performs pattern recall to retrieve a complete pattern from a partial or noisy input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Hopfield network-specific data to a binary writer. |
| `SetParameters(Vector<>)` |  |
| `Train(List<Vector<>>)` | Trains the Hopfield network on a set of patterns. |
| `Train(Tensor<>,Tensor<>)` | Trains the Hopfield network using the provided input patterns. |
| `UpdateParameters(Vector<>)` | Not implemented for Hopfield networks, as they don't use gradient-based parameter updates. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activationFunction` | Gets the activation function used to determine the state of each neuron. |
| `_size` | Gets or sets the size of the network, which is the number of neurons. |
| `_weights` | Gets or sets the weight matrix that stores the connection strengths between neurons. |

