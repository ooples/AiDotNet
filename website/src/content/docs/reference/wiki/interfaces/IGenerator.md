---
title: "IGenerator<T>"
description: "Defines the contract for text generation models used in retrieval-augmented generation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for text generation models used in retrieval-augmented generation.

## For Beginners

A generator is like a smart writer that creates answers.

Think of it like a research assistant:

- You ask a question: "What is machine learning?"
- The assistant reads relevant documents you provide
- The assistant writes an answer based on those documents
- The assistant includes references to show where information came from

In RAG systems:

1. Retriever finds relevant documents (research phase)
2. Generator reads those documents and writes the answer (writing phase)
3. The answer is "grounded" because it's based on real documents, not imagination

For example:

- Question: "How do transformers work?"
- Retrieved docs: 3 papers about transformer architecture
- Generated answer: "Transformers use self-attention mechanisms [1] to process

sequences in parallel [2], making them efficient for NLP tasks [3]."

- Citations [1], [2], [3] point to the source documents

## How It Works

A generator produces text responses based on input prompts, optionally augmented with
retrieved context. In RAG systems, generators take the user's query along with relevant
document snippets and produce grounded answers that cite their sources. The interface
extends IModel to integrate with the broader AiDotNet ecosystem.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxContextTokens` | Gets the maximum number of tokens this generator can process in a single request. |
| `MaxGenerationTokens` | Gets the maximum number of tokens this generator can generate in a response. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Generate(String)` | Generates a text response based on a prompt. |
| `GenerateGrounded(String,IEnumerable<Document<>>)` | Generates a grounded answer using provided context documents. |

