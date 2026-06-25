---
title: "Transformer<T>"
description: "Represents a Transformer neural network architecture, which is particularly effective for sequence-based tasks like natural language processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Transformer neural network architecture, which is particularly effective for
sequence-based tasks like natural language processing.

## For Beginners

A Transformer is a modern type of neural network that excels at 
understanding sequences of data, like sentences or time series.

Think of it like reading a book:

- When you read a sentence, some words are more important than others for understanding the meaning
- A Transformer can "pay attention" to different words based on their importance
- It can look at the entire context at once, rather than reading one word at a time

For example, in the sentence "The animal didn't cross the street because it was too wide",
the Transformer can figure out that "it" refers to "the street" by paying attention to the
relationship between these words.

Transformers are behind many recent AI advances, including large language models like GPT and BERT.

## How It Works

The Transformer architecture is a type of neural network design that uses self-attention mechanisms
instead of recurrence or convolution. This approach allows the model to weigh the importance of 
different parts of the input sequence when producing each part of the output sequence.

The key components of a Transformer include:

- Multi-head attention layers: Allow the model to focus on different parts of the input
- Feed-forward networks: Process the attended information
- Layer normalization: Stabilize the network during training
- Residual connections: Help information flow through the network

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Transformer(TransformerArchitecture<>,ILossFunction<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,TransformerOptions)` | Creates a new Transformer neural network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionMask` | Gets or sets the attention mask used in the Transformer. |
| `AuxiliaryLossWeight` | Gets or sets the weight for the attention regularization auxiliary loss. |
| `UseAuxiliaryLoss` | Gets or sets whether auxiliary loss (attention regularization) should be used during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for attention regularization across all attention layers. |
| `CreateDefaultVaswaniOptimizer` | Constructs the Vaswani 2017 §5.3 default optimizer recipe used throughout this class — Adam (β₁=0.9, β₂=0.98, ε=1e-9, lr=1e-3 sentinel) wrapped around a NoamSchedule bound to this instance's ModelDimension / WarmupSteps and stepped per batc… |
| `CreateNewInstance` | Creates a new instance of the Transformer with the same architecture and configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Transformer-specific data from a binary stream. |
| `EnumerateLayersAndSubLayers` | Enumerates every layer plus the registered sublayers of composite layers (one level deep — the encoder/decoder blocks the default factory emits). |
| `ForwardForTraining(Tensor<>)` | Training forward. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the attention regularization auxiliary loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Retrieves metadata about the Transformer model. |
| `GetOptions` |  |
| `InitializeLayers` | Sets up the layers of the Transformer network based on the provided architecture. |
| `PredictEager(Tensor<>)` | Performs a forward pass through the Transformer network to generate predictions. |
| `RunCheckpointedLayerWalk(Tensor<>,Int32)` | Gradient-checkpointed variant of `Tensor{` for encoder-decoder (and decoder-only) stacks. |
| `RunLayerWalk(Tensor<>)` | The transformer's sequential layer walk with encoder→decoder context routing. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Transformer-specific data to a binary stream. |
| `SetAttentionMask(Tensor<>)` | Sets the attention mask for the Transformer. |
| `SetBaseTrainOptimizer(IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Overrides the base hook so any caller of `Tensor{` — typically `AiModelBuilder.ConfigureOptimizer` — also updates the subclass-private `_optimizer` field. |
| `Train(Tensor<>,Tensor<>)` | Trains the Transformer on a single sample *or* a batched tensor of samples. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the Transformer network. |
| `ValidateCustomLayers(List<ILayer<>>)` | Ensures that custom layers provided for the Transformer are shape-compatible. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_optimizer` | The optimizer used to update the Transformer's parameters during training. |
| `_transformerArchitecture` | The configuration settings for this Transformer network. |

