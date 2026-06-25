---
title: "RecurrentNeuralNetwork<T>"
description: "Represents a Recurrent Neural Network, which is a type of neural network designed to process sequential data by maintaining an internal state."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Recurrent Neural Network, which is a type of neural network designed to process sequential data by maintaining an internal state.

## For Beginners

A Recurrent Neural Network is a type of neural network that has memory.

Think of it like reading a book:

- A standard neural network looks at each word in isolation
- An RNN remembers what it read earlier to understand the current word

For example, in the sentence "The clouds were dark, so I took my umbrella," an RNN understands that
"I took my umbrella" is related to "The clouds were dark" because it remembers the earlier part of the sentence.

This memory makes RNNs good at:

- Processing sequences like sentences, time series, or music
- Understanding context and relationships over time
- Predicting what comes next in a sequence

Common applications include text generation, translation, speech recognition, and stock price prediction - 
all tasks where what happened before affects how you interpret what's happening now.

## How It Works

A Recurrent Neural Network (RNN) is a class of neural networks that can process sequential data by maintaining
an internal state (memory) that captures information about previous inputs. Unlike traditional feedforward
neural networks, RNNs have connections that form directed cycles, allowing information to persist from one step
to the next. This architectural feature makes RNNs particularly well-suited for tasks involving sequential data,
such as time series prediction, natural language processing, and speech recognition.

**Reference:** Elman, J.L. (1990). "Finding structure in time." Cognitive Science, 14(2), 179-211.
https://doi.org/10.1016/0364-0213(90)90002-E

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RecurrentNeuralNetwork` | Initializes a new instance of the `RecurrentNeuralNetwork` class with the specified architecture. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the recurrent neural network with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data for the Recurrent Neural Network. |
| `GetModelMetadata` | Updates the parameters of all layers in the network based on computed gradients. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers based on the provided architecture or default configuration. |
| `PredictCore(Tensor<>)` | Makes a prediction using the current state of the Recurrent Neural Network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data for the Recurrent Neural Network. |
| `Train(Tensor<>,Tensor<>)` | Trains the Recurrent Neural Network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the recurrent neural network layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_learningRate` | The learning rate used for updating the network parameters during training. |

