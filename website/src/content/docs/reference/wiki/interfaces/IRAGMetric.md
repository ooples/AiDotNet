---
title: "IRAGMetric<T>"
description: "Defines the contract for RAG evaluation metrics."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for RAG evaluation metrics.

## For Beginners

Metrics are like test scores for your RAG system.

Think of it like grading an exam:

- The metric looks at the AI's answer
- Compares it to what the answer should be (or checks quality)
- Gives a score (0-1, where 1 is perfect)

Different metrics measure different things:

- Faithfulness: Does the answer stick to the source documents?
- Similarity: How close is the answer to the ground truth?
- Coverage: Does the answer address all parts of the question?

Use metrics to:

- Compare different RAG configurations
- Track improvements over time
- Identify weak points in your system

## How It Works

A RAG metric evaluates the quality of retrieval-augmented generation systems
by comparing generated answers against ground truth or analyzing specific aspects
of the generation process. Metrics help developers understand system performance
and guide improvements.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the description of what this metric measures. |
| `Name` | Gets the name of this metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(GroundedAnswer<>,String)` | Evaluates a grounded answer and returns a score. |

