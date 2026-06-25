---
title: "InstructorEmbedding<T>"
description: "Instructor/E5 (Instruction-Tuned) embedding model implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Instructor/E5 (Instruction-Tuned) embedding model implementation.
Uses task-specific instructions to adapt embeddings for different use cases.

## For Beginners

Most AI models read every sentence the same way. "Instructor" models are 
like specialized scholars. If you tell them "read this like a doctor looking for a diagnosis," 
they will focus on medical terms. If you tell them "read this like a poet," they will focus 
on the mood. It makes the "coordinates" (embeddings) much more useful for your specific goal.

## How It Works

Instructor models are transformer-based encoders trained with instructions. By prepending a task 
description (e.g., "Represent the Wikipedia sentence for retrieval:"), the model learns to 
produce embeddings that are optimized for that specific task.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstructorEmbedding` | Initializes a new instance with default architecture settings. |
| `InstructorEmbedding(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32,Int32,Int32,Int32,TransformerEmbeddingNetwork<>.PoolingStrategy,ILossFunction<>,Double,InstructorEmbeddingOptions)` | Initializes a new instance of the InstructorEmbedding model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Embed(String)` |  |
| `EmbedAsync(String)` |  |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `EmbedWithInstruction(String,String)` | Encodes text into a normalized embedding vector using a task-specific instruction. |
| `GetModelMetadata` | Retrieves metadata about the Instructor model, including its default instruction. |
| `GetOptions` |  |
| `InitializeLayers` | Configures the transformer encoder layers for the Instructor architecture. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetDefaultInstruction(String)` | Sets the default instruction used for general embedding generation. |

