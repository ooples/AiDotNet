---
title: "ContextCompressorBase<T>"
description: "Provides a base implementation for context compressors with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.ContextCompression`

Provides a base implementation for context compressors with common functionality.

## For Beginners

This is the foundation that all context compressors build upon.

Think of it like a template for reducing document size:

- It handles common tasks (checking inputs aren't null or empty)
- Specific compression methods (LLM-based, rule-based) fill in how they compress
- This ensures all compressors work consistently

## How It Works

This abstract class implements the IContextCompressor interface and provides common functionality
for context compression strategies. It handles validation and delegates to derived classes
for the core compression logic.

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(List<Document<>>,String,Dictionary<String,Object>)` | Compresses a collection of documents while preserving relevance to the query. |
| `CompressCore(List<Document<>>,String,Dictionary<String,Object>)` | Core compression logic to be implemented by derived classes. |
| `ValidateDocuments(List<Document<>>)` | Validates the document collection. |
| `ValidateQuery(String)` | Validates the query string. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

