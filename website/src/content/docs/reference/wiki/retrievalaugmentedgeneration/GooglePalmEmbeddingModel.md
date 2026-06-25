---
title: "GooglePalmEmbeddingModel<T>"
description: "Google PaLM embedding model integration via Vertex AI."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels`

Google PaLM embedding model integration via Vertex AI.

## How It Works

Provides access to Google's PaLM (Pathways Language Model) and Gemini embedding capabilities
through the Google Cloud Vertex AI platform.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GooglePalmEmbeddingModel(String,String,String,String,Int32,HttpClient)` | Initializes a new instance of the `GooglePalmEmbeddingModel` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedAsync(String)` | Asynchronously generates an embedding for the specified text using Google Vertex AI API. |
| `EmbedBatchAsync(IEnumerable<String>)` | Asynchronously generates embeddings for the specified texts using Google Vertex AI API. |
| `EmbedCore(String)` | Generates embeddings using Google Vertex AI API. |

