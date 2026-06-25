---
title: "VectorDocument<T>"
description: "Represents a document paired with its vector embedding for storage and retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Models`

Represents a document paired with its vector embedding for storage and retrieval.

## For Beginners

A VectorDocument is like a book with its catalog card.

Think of it as two pieces working together:

- Document: The actual book (content, title, author, etc.)
- Embedding: The numerical "fingerprint" describing what the book is about

Why combine them?
When you add documents to a search system, you need both:

- The vector (for finding similar documents through math)
- The document (for returning the actual content to users)

For example:

- Document: "Climate change affects global temperatures..."
- Embedding: [0.23, -0.45, 0.78, ..., 0.12] (768 numbers)

The system uses the numbers to search, then returns the text.

## How It Works

A VectorDocument combines a Document with its vector embedding, creating a complete
unit ready for indexing in a vector store. The vector embedding captures the semantic
meaning of the document's content in a numerical form suitable for similarity calculations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VectorDocument` | Initializes a new instance of the VectorDocument class. |
| `VectorDocument(Document<>,Vector<>)` | Initializes a new instance of the VectorDocument class with a document and embedding. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Document` | Gets or sets the document containing the text content and metadata. |
| `Embedding` | Gets or sets the vector embedding representing the document's semantic meaning. |

