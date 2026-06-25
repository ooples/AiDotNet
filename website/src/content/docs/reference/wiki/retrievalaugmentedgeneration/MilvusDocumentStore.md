---
title: "MilvusDocumentStore<T>"
description: "MilvusDocumentStore<T> — Models & Types in AiDotNet.RetrievalAugmentedGeneration.DocumentStores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.DocumentStores`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MilvusDocumentStore(String,Int32)` | Initializes a new instance of the MilvusDocumentStore class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DocumentCount` | Gets the number of documents currently stored in this collection. |
| `VectorDimension` | Gets the dimensionality of vectors stored in this collection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBatchCore(IList<VectorDocument<>>)` | Core logic for adding multiple vector documents in a batch operation. |
| `AddCore(VectorDocument<>)` | Core logic for adding a single vector document to the collection. |
| `Clear` | Removes all documents from the collection and resets vector dimension. |
| `GetAllCore` | Core logic for retrieving all documents in the collection. |
| `GetByIdCore(String)` | Core logic for retrieving a document by its unique identifier. |
| `GetSimilarCore(Vector<>,Int32,Dictionary<String,Object>)` | Core logic for similarity search with optional metadata filtering. |
| `RemoveCore(String)` | Core logic for removing a document from the collection. |

