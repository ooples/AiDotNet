---
title: "Document<T>"
description: "Represents a document with content, metadata, and optional relevance scoring."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Models`

Represents a document with content, metadata, and optional relevance scoring.

## For Beginners

A document is like a file or article in your system.

Think of it like a book entry in a library catalog:

- Id: The unique catalog number
- Content: The actual text from the book
- Metadata: Information about the book (author, date, category, etc.)
- RelevanceScore: How well this book matches what you're looking for

For example, when you search for "climate change", documents about environmental
science get high relevance scores, while documents about sports get low scores.

## How It Works

A document is the fundamental unit of information in a retrieval-augmented generation system.
It contains the actual text content, metadata for filtering and tracking, and optional
relevance scores assigned during retrieval or reranking processes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Document` | Initializes a new instance of the Document class. |
| `Document(String,String)` | Initializes a new instance of the Document class with specified content. |
| `Document(String,String,Dictionary<String,Object>)` | Initializes a new instance of the Document class with content and metadata. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Content` | Gets or sets the text content of the document. |
| `Embedding` | Gets or sets the embedding vector for this document. |
| `HasRelevanceScore` | Gets or sets whether this document has a relevance score assigned. |
| `Id` | Gets or sets the unique identifier for this document. |
| `Metadata` | Gets or sets metadata associated with this document. |
| `RelevanceScore` | Gets or sets the relevance score assigned to this document by a retriever or reranker. |

