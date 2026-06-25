---
title: "DocumentStoreBase<T>"
description: "Provides a base implementation for document stores with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.DocumentStores`

Provides a base implementation for document stores with common functionality.

## For Beginners

This is the foundation that all document storage systems build upon.

Think of it like a template for building a library:

- It handles common tasks (checking inputs, managing documents, calculating similarity)
- Specific storage systems (in-memory, database) just fill in where/how documents are stored
- This ensures all document stores work consistently

## How It Works

This abstract class implements the IDocumentStore interface and provides common functionality
for vector document storage and retrieval. It handles validation, document management, and
provides utility methods for similarity calculations while allowing derived classes to focus
on implementing storage-specific logic and search algorithms.

## Properties

| Property | Summary |
|:-----|:--------|
| `DocumentCount` | Gets the number of documents currently stored in the document store. |
| `VectorDimension` | Gets the dimensionality of the vectors stored in this document store. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(VectorDocument<>)` | Adds a single vectorized document to the store. |
| `AddBatch(IEnumerable<VectorDocument<>>)` | Adds multiple vectorized documents to the store in a batch operation. |
| `AddBatchCore(IList<VectorDocument<>>)` | Core logic for adding multiple vector documents in a batch. |
| `AddCore(VectorDocument<>)` | Core logic for adding a single vector document. |
| `Clear` | Removes all documents from the store. |
| `GetAll` | Gets all documents currently stored in the document store. |
| `GetAllCore` | Core logic for retrieving all documents. |
| `GetById(String)` | Retrieves a document by its unique identifier. |
| `GetByIdCore(String)` | Core logic for retrieving a document by ID. |
| `GetSimilar(Vector<>,Int32)` | Retrieves the top-k most similar documents to a given query vector. |
| `GetSimilarCore(Vector<>,Int32,Dictionary<String,Object>)` | Core logic for similarity search with optional filtering. |
| `GetSimilarWithFilters(Vector<>,Int32,Dictionary<String,Object>)` | Retrieves similar documents with additional metadata filtering. |
| `MatchesFilters(Document<>,Dictionary<String,Object>)` | Checks if a document matches the specified metadata filters. |
| `Remove(String)` | Removes a document from the store by its identifier. |
| `RemoveCore(String)` | Core logic for removing a document by ID. |
| `ValidateDocumentId(String)` | Validates a document ID. |
| `ValidateMetadataFilters(Dictionary<String,Object>)` | Validates metadata filters. |
| `ValidateQueryVector(Vector<>)` | Validates a query vector. |
| `ValidateTopK(Int32)` | Validates the topK parameter. |
| `ValidateVectorDocument(VectorDocument<>)` | Validates a vector document before adding it to the store. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

