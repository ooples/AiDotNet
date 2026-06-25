---
title: "GRUNeuralNetwork<T>"
description: "Represents a Gated Recurrent Unit (GRU) Neural Network for processing sequential data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Gated Recurrent Unit (GRU) Neural Network for processing sequential data.

## For Beginners

A GRU Neural Network is a special type of neural network that's good at processing data that comes in sequences.

Think of it like reading a book:

- A regular neural network would look at each word in isolation
- A GRU network remembers what it read earlier and uses that context to understand each new word

GRU networks have special "gates" that control what information to remember and what to forget:

- This helps them understand patterns that stretch over long sequences
- For example, in a sentence like "John went to the store because he needed milk," a GRU can connect "he" with "John"

GRU networks are useful for:

- Text processing and generation
- Time series prediction (like stock prices or weather)
- Speech recognition
- Any task where the order and context of data matters

## How It Works

A GRU Neural Network is a type of recurrent neural network designed to effectively model sequential data.
GRU networks use gating mechanisms to control information flow through the network, allowing them to
capture long-term dependencies in sequence data while avoiding the vanishing gradient problem
that affects simple recurrent networks.

**Reference:** Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder
for Statistical Machine Translation" (EMNLP 2014). ``

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GRUNeuralNetwork` | Initializes a new instance of the `GRUNeuralNetwork` class with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the GRU Neural Network with the same architecture and configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Loads GRU-specific data from a binary stream. |
| `ForwardWithMemory(Tensor<>)` | Performs a forward pass while storing intermediate values for backpropagation. |
| `GetModelMetadata` | Gets metadata about this GRU Neural Network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Performs a forward pass through the network and generates predictions. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Saves GRU-specific data to a binary stream. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network using the provided parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_trainOptimizer` | Trains the GRU network using the provided input and expected output. |

