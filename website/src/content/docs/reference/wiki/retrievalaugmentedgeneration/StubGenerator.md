---
title: "StubGenerator<T>"
description: "A simple stub generator for testing and development that creates template-based answers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Generators`

A simple stub generator for testing and development that creates template-based answers.

## For Beginners

This is a simple placeholder until real LLM generators are ready.

Think of it like an auto-reply email:

- It doesn't actually understand the question
- It just formats the retrieved documents into an answer
- Adds citation numbers [1], [2], [3]
- Good enough for testing the RAG pipeline
- Replace with a real LLM (GPT, Claude, etc.) for production

For example:

- Question: "What is photosynthesis?"
- Retrieved docs: 3 biology documents
- Generated answer: "Based on the provided context: [Document 1 content] [1].

[Document 2 content] [2]. [Document 3 content] [3]."

Not intelligent, but proves the pipeline works!
This enables development on Issue #284 without waiting for transformer integration.

## How It Works

This implementation creates simple grounded answers by concatenating context documents
with basic citation markers. It's designed for testing the RAG pipeline structure before
real generation models are integrated. The generator uses a template-based approach to
create answers that include numbered citations to source documents.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StubGenerator(Int32,Int32)` | Initializes a new instance of the StubGenerator class. |

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

