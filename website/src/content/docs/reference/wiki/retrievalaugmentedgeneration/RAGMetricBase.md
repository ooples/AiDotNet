---
title: "RAGMetricBase<T>"
description: "Provides a base implementation for RAG evaluation metrics with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.Evaluation`

Provides a base implementation for RAG evaluation metrics with common functionality.

## For Beginners

This is the foundation for all RAG metrics.

It handles common tasks like:

- Validating inputs (checking for null values)
- Normalizing scores (ensuring they're between 0 and 1)
- Providing helper methods for common calculations

Specific metrics (Faithfulness, Similarity, etc.) just need to implement
their specific scoring logic.

## How It Works

This abstract class implements the IRAGMetric interface and provides common validation
and utility methods for metric implementations. It ensures consistent behavior across
different metrics while allowing derived classes to focus on specific evaluation logic.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the description of what this metric measures. |
| `Name` | Gets the name of this metric. |
| `RequiresGroundTruth` | Gets a value indicating whether this metric requires ground truth for evaluation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(GroundedAnswer<>,String)` | Evaluates a grounded answer and returns a score. |
| `EvaluateCore(GroundedAnswer<>,String)` | Core evaluation logic to be implemented by derived classes. |
| `GetWords(String)` | Extracts words from text. |
| `ValidateAnswer(GroundedAnswer<>)` | Validates the grounded answer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

