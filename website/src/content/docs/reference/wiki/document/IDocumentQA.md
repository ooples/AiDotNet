---
title: "IDocumentQA<T>"
description: "Interface for document question answering models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Interface for document question answering models.

## For Beginners

Document QA is like having a smart assistant that can read
a document and answer your questions about it. You show it a document image and
ask questions like "What is the total amount?" or "Who signed this contract?"

Example usage:

## How It Works

Document QA models answer natural language questions about document content,
combining visual understanding with text comprehension.

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` | Answers a question about a document. |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` | Answers a question with generation parameters. |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` | Answers multiple questions about a document in a batch. |
| `ExtractFields(Tensor<>,IEnumerable<String>)` | Extracts specific fields from a document using natural language prompts. |

