---
title: "PostgresVectorDocumentStore<T>"
description: "PostgresVectorDocumentStore<T> — Models & Types in AiDotNet.RetrievalAugmentedGeneration.DocumentStores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.DocumentStores`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PostgresVectorDocumentStore(String,Int32)` | Initializes a new instance of the PostgresVectorDocumentStore class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DocumentCount` | Gets the number of documents currently stored in the table. |
| `VectorDimension` | Gets the dimensionality of vectors stored in this table. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBatchCore(IList<VectorDocument<>>)` | Core logic for adding multiple vector documents in a batch operation. |
| `AddCore(VectorDocument<>)` | Core logic for adding a single vector document to the table. |
| `Clear` | Removes all documents from the table and resets the vector dimension. |
| `GetAllCore` | Core logic for retrieving all documents in the table. |
| `GetByIdCore(String)` | Core logic for retrieving a document by its unique identifier. |
| `RemoveCore(String)` | Core logic for removing a document from the table. |

