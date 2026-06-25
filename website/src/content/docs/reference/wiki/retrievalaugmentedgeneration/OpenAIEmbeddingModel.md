---
title: "OpenAIEmbeddingModel<T>"
description: "OpenAI embedding model for generating embeddings via OpenAI API."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels`

OpenAI embedding model for generating embeddings via OpenAI API.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenAIEmbeddingModel(String,String,Int32,Int32,HttpClient)` | Initializes a new instance of the `OpenAIEmbeddingModel` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedAsync(String)` | Asynchronously encodes a single text into a vector representation via the OpenAI API. |
| `EmbedBatchAsync(IEnumerable<String>)` | Asynchronously encodes a batch of texts into vector representations via the OpenAI API. |

