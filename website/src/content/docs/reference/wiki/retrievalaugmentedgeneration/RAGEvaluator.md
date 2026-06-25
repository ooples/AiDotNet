---
title: "RAGEvaluator<T>"
description: "Evaluates RAG system performance using multiple metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Evaluation`

Evaluates RAG system performance using multiple metrics.

## For Beginners

This runs all your tests on the RAG system and gives you a report card.

Think of it like grading a student across multiple subjects:

- Math (Faithfulness): Did they show their work correctly?
- English (Similarity): How close is the essay to the example?
- Science (Coverage): Did they research enough sources?

The evaluator:

1. Takes your RAG system's answer
2. Runs all configured metrics on it
3. Gives you scores for each metric
4. Calculates an overall average score

Use this to:

- Compare different RAG configurations
- Track improvements over time
- Identify specific weaknesses
- Make data-driven optimization decisions

## How It Works

The RAG evaluator runs multiple evaluation metrics on grounded answers and aggregates
the results. This provides a comprehensive view of RAG system performance across different
quality dimensions (faithfulness, similarity, coverage, etc.).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RAGEvaluator` | Initializes a new instance of the RAGEvaluator class with default metrics. |
| `RAGEvaluator(IEnumerable<IRAGMetric<>>)` | Initializes a new instance of the RAGEvaluator class with specified metrics. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metrics` | Gets the metrics used by this evaluator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(GroundedAnswer<>,String)` | Evaluates a grounded answer using all configured metrics. |
| `EvaluateBatch(IEnumerable<GroundedAnswer<>>,IEnumerable<String>)` | Evaluates multiple grounded answers and returns aggregated results. |
| `GetAggregateStats(IEnumerable<EvaluationResult>)` | Calculates aggregate statistics across multiple evaluation results. |

