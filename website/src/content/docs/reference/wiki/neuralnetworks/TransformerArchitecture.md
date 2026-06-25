---
title: "TransformerArchitecture<T>"
description: "Defines the architecture configuration for a Transformer neural network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Defines the architecture configuration for a Transformer neural network.

## For Beginners

Think of this class as a blueprint for building a Transformer.

Just like building a house requires decisions about how many rooms, how big each room should be, 
and what materials to use, building a Transformer requires decisions about:

- How many layers of processing to include
- How much information to process at once
- How to connect different parts of the network

This class stores all those decisions in one place, making it easier to create Transformer
networks with different capabilities for different tasks.

## How It Works

The TransformerArchitecture class encapsulates all the hyperparameters and configuration options
needed to define a Transformer neural network. It includes settings for the encoder and decoder stacks,
attention mechanisms, model dimensions, and other key aspects that determine the network's structure
and behavior.

Transformers are particularly effective for sequence-based tasks like natural language processing,
translation, text summarization, and other tasks that benefit from understanding the relationships
between elements in a sequence.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerArchitecture(InputType,NeuralNetworkTaskType,Int32,Int32,Int32,Int32,Int32,NetworkComplexity,Int32,Int32,Double,Int32,Int32,Boolean,Double,Nullable<SequencePoolingMode>,List<ILayer<>>)` | Initializes a new instance of the `TransformerArchitecture` class with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets the dropout rate used for regularization in the Transformer. |
| `FeedForwardDimension` | Gets the dimension of the feed-forward networks within the Transformer layers. |
| `MaxSequenceLength` | Gets the maximum length of input sequences that the Transformer can process. |
| `ModelDimension` | Gets the dimension of the model's internal representations. |
| `NumDecoderLayers` | Gets the number of decoder layers in the Transformer. |
| `NumEncoderLayers` | Gets the number of encoder layers in the Transformer. |
| `NumHeads` | Gets the number of attention heads in each multi-head attention layer. |
| `SequencePooling` | Strategy for collapsing the encoder's `[batch, seq, dim]` hidden states into a single `[batch, dim]` vector before the classification head, when the task is single-label per sequence. |
| `Temperature` | Gets the temperature parameter used for controlling randomness in text generation. |
| `UsePositionalEncoding` | Gets a value indicating whether positional encoding is used in the Transformer. |
| `VocabularySize` | Gets the size of the vocabulary for text-based tasks. |
| `WarmupSteps` | Gets the number of warmup steps for the default Noam (Vaswani 2017) learning-rate schedule attached to the Transformer's default Adam optimizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InferClassificationTaskType(Int32[],Int32)` | Infers the correct classification task type from the shape of the target data. |
| `ValidateTaskTypeVsTargetShape(NeuralNetworkTaskType,Int32[],Int32)` | Validates that the configured task type is consistent with the target data shape. |

