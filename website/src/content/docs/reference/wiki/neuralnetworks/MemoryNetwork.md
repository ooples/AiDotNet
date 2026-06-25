---
title: "MemoryNetwork<T>"
description: "Represents a Memory Network, a neural network architecture designed with explicit memory components for improved reasoning and question answering capabilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Memory Network, a neural network architecture designed with explicit memory components
for improved reasoning and question answering capabilities.

## For Beginners

A Memory Network is a special type of neural network that has its own "memory storage" component.

Think of it like a person who has:

- A notebook (the memory) where they can write down important facts
- The ability to read specific information from their notebook when needed
- The ability to add new information to their notebook as they learn

For example, if you provided a Memory Network with several facts about a topic:

- It would store these facts in its memory matrix
- When asked a question, it would search its memory for relevant information
- It would use this retrieved information to formulate an answer

Memory Networks are particularly good at:

- Question answering based on a given context
- Reasoning tasks that require remembering multiple facts
- Dialog systems that need to maintain conversation history
- Tasks where information needs to be remembered and used later

## How It Works

Memory Networks combine neural network components with a long-term memory matrix that can be read from
and written to. This architecture allows the network to store information persistently and access it
selectively when needed, making it particularly effective for tasks requiring reasoning over facts or
answering questions based on provided context.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryNetwork` | Initializes a new instance of the `MemoryNetwork` class with the specified architecture and memory parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>)` | Queries the memory network with a question and returns the answer. |
| `ApplySoftmax(Tensor<>)` | Applies softmax normalization to attention logits. |
| `AreShapesCompatible(Int32[],Int32[])` | Checks if two tensor shapes are compatible for element-wise operations. |
| `CalculateAttention(Tensor<>)` | Calculates attention weights over memory slots based on the encoded input. |
| `CalculateMeanSquaredError(Tensor<>,Tensor<>)` | Calculates mean squared error between predictions and expected outputs. |
| `CalculateOutputGradients(Tensor<>,Tensor<>)` | Calculates gradients for output layer based on predictions and expected outputs. |
| `CombineInputAndMemory(Tensor<>,Tensor<>)` | Combines the encoded input with the memory readout. |
| `CreateNewInstance` | Creates a new instance of the Memory Network with the same architecture and configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes memory network-specific data from a binary reader. |
| `EncodeInput(Tensor<>)` | Encodes the input using the input encoding layers. |
| `ForwardForTraining(Tensor<>)` |  |
| `GenerateOutput(Tensor<>)` | Generates the final output from the combined representation. |
| `GetModelMetadata` | Gets metadata about the memory network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the Memory Network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Processes input through the memory network to generate predictions. |
| `ReadFromMemory(Tensor<>)` | Reads from memory using attention weights. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes memory network-specific data to a binary writer. |
| `ShouldUpdateMemoryDuringInference` | Determines whether memory should be updated during inference. |
| `StoreFact(Tensor<>)` | Stores a new fact in memory. |
| `UpdateMemory(Tensor<>,Tensor<>)` | Updates memory with new information. |
| `UpdateMemoryNetworkParameters` | Updates the memory network parameters based on calculated gradients. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network using the provided parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingSize` | Gets the size of each memory embedding vector. |
| `_memory` | Gets or sets the memory matrix that stores embeddings. |
| `_memorySize` | Gets the size of the memory (number of memory slots). |
| `_trainOptimizer` | Trains the memory network on input-output pairs. |

