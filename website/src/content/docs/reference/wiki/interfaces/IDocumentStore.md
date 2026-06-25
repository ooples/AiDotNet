---
title: "IDocumentStore<T>"
description: "Defines the contract for document stores that index and retrieve vectorized documents."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for document stores that index and retrieve vectorized documents.

## For Beginners

A document store is like a smart library that organizes information by meaning.

Think of it like a special filing cabinet:

- Regular filing cabinet: Organized alphabetically or by date
- Document store: Organized by *meaning* using math

When you search for "climate change", it finds documents about environmental issues
even if they don't contain those exact words, because it understands the *meaning*.

It's like having a librarian who truly understands what each book is about and can
find exactly what you need based on your question, not just keywords.

## How It Works

A document store manages a collection of documents with their vector embeddings,
enabling efficient similarity-based retrieval. Implementations can range from simple
in-memory storage to distributed vector databases. The interface supports adding documents,
similarity search, and metadata-based filtering.

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
| `Clear` | Removes all documents from the store. |
| `GetAll` | Gets all documents currently stored in the document store. |
| `GetById(String)` | Retrieves a document by its unique identifier. |
| `GetSimilar(Vector<>,Int32)` | Retrieves the top-k most similar documents to a given query vector. |
| `GetSimilarWithFilters(Vector<>,Int32,Dictionary<String,Object>)` | Retrieves similar documents with additional metadata filtering. |
| `Remove(String)` | Removes a document from the store by its identifier. |

