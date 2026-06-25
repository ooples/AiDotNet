---
title: "HybridDocumentStore<T>"
description: "HybridDocumentStore<T> — Models & Types in AiDotNet.RetrievalAugmentedGeneration.DocumentStores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.DocumentStores`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridDocumentStore(IDocumentStore<>,IDocumentStore<>,,)` | Initializes a new instance of the HybridDocumentStore class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DocumentCount` | Gets the number of documents currently stored (from the vector store). |
| `VectorDimension` | Gets the dimensionality of vectors stored (from the vector store). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBatchCore(IList<VectorDocument<>>)` | Core logic for adding multiple vector documents to both underlying stores in a batch. |
| `AddCore(VectorDocument<>)` | Core logic for adding a single vector document to both underlying stores. |
| `Clear` | Removes all documents from both underlying stores. |
| `GetAllCore` | Core logic for retrieving all documents from the vector store. |
| `GetByIdCore(String)` | Core logic for retrieving a document by ID from the vector store. |
| `GetSimilarCore(Vector<>,Int32,Dictionary<String,Object>)` | Core logic for hybrid search combining vector similarity and keyword matching. |
| `RemoveCore(String)` | Core logic for removing a document from both underlying stores. |

