---
title: "ONNXSentenceTransformer<T>"
description: "Production-ready sentence transformer for generating semantic embeddings using ONNX Runtime."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels`

Production-ready sentence transformer for generating semantic embeddings using ONNX Runtime.

## Properties

| Property | Summary |
|:-----|:--------|
| `Session` | Gets the inference session, ensuring it's loaded. |
| `Tokenizer` | Gets the tokenizer, ensuring it's loaded. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildOnnxInputs(ICollection<String>,Int64[],Int64[],Int32[])` | Builds the list of ONNX inputs for a sentence-transformer encode call. |
| `EnsureModelLoaded` | Ensures the ONNX model and tokenizer are loaded, loading lazily on first use. |
| `GenerateFallbackEmbedding(String)` | Generates a deterministic fallback embedding based on the text hash. |

