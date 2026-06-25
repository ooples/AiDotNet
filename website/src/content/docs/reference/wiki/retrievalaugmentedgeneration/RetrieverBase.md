---
title: "RetrieverBase<T>"
description: "Provides a base implementation for document retrievers with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.Retrievers`

Provides a base implementation for document retrievers with common functionality.

## For Beginners

This is the foundation that all retrieval methods build upon.

Think of it like a template for search engines:

- It handles common tasks (checking inputs, limiting results, sorting)
- Specific retrieval methods (vector search, keyword search) just fill in how they find documents
- This ensures all retrievers work consistently

## How It Works

This abstract class implements the IRetriever interface and provides common functionality
for document retrieval strategies. It handles validation, result limiting, and post-processing
while allowing derived classes to focus on implementing the core retrieval algorithm.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RetrieverBase(Int32)` | Initializes a new instance of the RetrieverBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultTopK` | Gets the default number of documents to retrieve. |

## Methods

| Method | Summary |
|:-----|:--------|
| `PostProcessResults(IEnumerable<Document<>>,Int32)` | Post-processes retrieved results before returning them. |
| `Retrieve(String)` | Retrieves relevant documents for a given query string using the default TopK value. |
| `Retrieve(String,Int32)` | Retrieves relevant documents with a custom number of results. |
| `Retrieve(String,Int32,Dictionary<String,Object>)` | Retrieves relevant documents with metadata filtering. |
| `RetrieveCore(String,Int32,Dictionary<String,Object>)` | Core retrieval logic to be implemented by derived classes. |
| `ValidateMetadataFilters(Dictionary<String,Object>)` | Validates the metadata filters. |
| `ValidateQuery(String)` | Validates the query string. |
| `ValidateTopK(Int32)` | Validates the topK parameter. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

