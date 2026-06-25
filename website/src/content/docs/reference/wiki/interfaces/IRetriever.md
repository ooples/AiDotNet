---
title: "IRetriever<T>"
description: "Defines the contract for retrieving relevant documents based on a query."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for retrieving relevant documents based on a query.

## For Beginners

A retriever is like a smart search engine for your documents.

Think of it like different ways to find information:

- Dense retrieval: Understands meaning (finds "automobile" when you search "car")
- Sparse retrieval: Matches keywords (finds exact words you typed)
- Hybrid retrieval: Combines both approaches for best results

When you ask a question, the retriever finds the documents most likely to contain
the answer, even if they don't use the exact same words you used.

## How It Works

A retriever finds the most relevant documents for a given query using various
retrieval strategies such as dense vector search, sparse keyword matching, or
hybrid approaches. Implementations can range from simple vector similarity to
complex multi-stage retrieval pipelines.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultTopK` | Gets the default number of documents to retrieve. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Retrieve(String)` | Retrieves relevant documents for a given query string using the default TopK value. |
| `Retrieve(String,Int32)` | Retrieves relevant documents with a custom number of results. |
| `Retrieve(String,Int32,Dictionary<String,Object>)` | Retrieves relevant documents with metadata filtering. |

