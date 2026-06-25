---
title: "BGE<T>"
description: "BGE (BAAI General Embedding) neural network implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

BGE (BAAI General Embedding) neural network implementation.
A state-of-the-art retrieval model known for its high accuracy across diverse benchmarks.

## For Beginners

BGE is currently one of the "smartest" search engines in the world. It has been 
trained like a student who went through elementary school (general reading), high school (specific facts), 
and then a PhD program (answering hard questions). This makes it incredibly good at understanding 
exactly what you're looking for, even if your query is phrased in a confusing way.

## How It Works

BGE is a series of open-source embedding models from the Beijing Academy of Artificial Intelligence (BAAI). 
These models are specifically optimized for retrieval tasks using a multi-stage training curriculum 
that includes massive-scale pre-training and fine-grained instruction tuning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BGE` | Initializes a new instance of the BGE model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Embed(String)` |  |
| `EmbedAsync(String)` |  |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `GetModelMetadata` | Retrieves metadata about the BGE model. |
| `GetOptions` |  |
| `InitializeLayers` | Configures the transformer layers for the BGE model using optimized retrieval defaults from LayerHelper. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |

