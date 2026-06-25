---
title: "FAISSDocumentStore<T>"
description: "FAISSDocumentStore<T> — Models & Types in AiDotNet.RetrievalAugmentedGeneration.DocumentStores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.DocumentStores`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FAISSDocumentStore(Int32)` | Initializes a new instance of the FAISSDocumentStore class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DocumentCount` | Gets the number of documents currently stored. |
| `VectorDimension` | Gets the dimensionality of vectors stored in this index. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBatchCore(IList<VectorDocument<>>)` | Core logic for adding multiple vector documents in a batch with automatic indexing. |
| `AddCore(VectorDocument<>)` | Core logic for adding a single vector document with automatic indexing. |
| `Clear` | Removes all documents from the store and resets all indices. |
| `GetAllCore` | Core logic for retrieving all documents in the store. |
| `GetByIdCore(String)` | Core logic for retrieving a document by its unique identifier. |
| `GetSimilarCore(Vector<>,Int32,Dictionary<String,Object>)` | Core logic for similarity search using cosine similarity with optional metadata filtering. |
| `RemoveCore(String)` | Core logic for removing a document from the store. |

