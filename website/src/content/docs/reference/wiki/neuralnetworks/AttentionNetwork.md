---
title: "AttentionNetwork<T>"
description: "Represents a neural network that utilizes attention mechanisms for sequence processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a neural network that utilizes attention mechanisms for sequence processing.

## For Beginners

This network mimics how humans pay attention to different parts of information.

Think of it like reading a complex paragraph:

- When you try to understand a sentence, you don't focus equally on all words
- You focus more on the important words that carry meaning
- You also connect related words even if they're far apart

For example, in the sentence "The cat, which had a white spot on its tail, chased the mouse":

- An attention network would connect "cat" with "chased" even though they're separated
- It would assign different importance to different words based on context
- This helps it understand the overall meaning better than networks that process words in isolation

This ability to selectively focus and connect distant information makes attention networks
powerful for language tasks, time series prediction, and many other sequence-based problems.

## How It Works

An attention network is a specialized neural network architecture designed for sequence processing tasks.
It uses attention mechanisms to dynamically focus on different parts of the input sequence when generating
outputs. This allows the network to capture long-range dependencies and relationships between elements in 
the sequence, making it particularly effective for tasks like natural language processing, time series analysis,
and other sequence-to-sequence problems.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AttentionNetwork` | Initializes a new instance with default architecture settings. |
| `AttentionNetwork(NeuralNetworkArchitecture<>,Int32,Int32,ILossFunction<>,AttentionNetworkOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `AttentionNetwork` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for attention entropy regularization. |
| `UseAuxiliaryLoss` | Gets or sets whether to use auxiliary loss (attention entropy regularization) during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for the AttentionNetwork, which aggregates attention entropy losses from all attention layers. |
| `CreateNewInstance` | Creates a new instance of the attention network model. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data for the Attention Network. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the attention entropy regularization. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Gets metadata about the Attention Network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the attention network. |
| `PredictCore(Tensor<>)` | Makes a prediction using the current state of the Attention Network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data for the Attention Network. |
| `Train(Tensor<>,Tensor<>)` | Trains the Attention Network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the attention network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingSize` | The size of the embeddings used to represent each element in the sequence. |
| `_lastAttentionEntropyLoss` | Stores the last computed attention entropy loss for diagnostics. |
| `_lossFunction` | The loss function used to measure the network's performance during training. |
| `_optimizer` | The optimizer used for parameter updates during training. |
| `_sequenceLength` | The maximum length of sequences this network can process. |

