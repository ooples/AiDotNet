---
title: "ContextCoverageMetric<T>"
description: "Evaluates how well the retrieved documents cover the information needed to answer the query."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Evaluation`

Evaluates how well the retrieved documents cover the information needed to answer the query.

## For Beginners

This checks if the retrieved documents have enough info to answer the question.

Think of it like checking if you brought the right textbooks to class:

- Question: "What is photosynthesis?"
- Retrieved docs about plants, sunlight, energy ✓ (good coverage)
- Retrieved docs about animals, water, soil ✗ (poor coverage)

How it works:

**With Ground Truth** (you know the correct answer):

- Checks if the retrieved documents contain words from the correct answer
- High score: Sources have all the key information
- Low score: Sources are missing important facts

**Without Ground Truth** (no reference answer):

- Uses relevance scores from retrieval
- Checks document diversity (not all the same topic)
- Estimates if sources are comprehensive enough

For example:

- Query: "What are the products of photosynthesis?"
- Ground truth: "Glucose and oxygen"
- Retrieved docs mention "glucose" and "oxygen" ✓ (score: 1.0)
- Retrieved docs only mention "glucose" ✗ (score: 0.5)

Why this matters:

- Bad retrieval = bad answers (garbage in, garbage out)
- Identifies when you need better retrieval or more documents
- Helps tune retrieval parameters (topK, similarity threshold)

## How It Works

Context coverage measures whether the retrieved documents contain the information needed
to answer the query. When ground truth is provided, it checks if the sources contain
the key information from the correct answer. Without ground truth, it estimates coverage
by analyzing document relevance scores and diversity.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the description of what this metric measures. |
| `Name` | Gets the name of this metric. |
| `RequiresGroundTruth` | Gets a value indicating whether this metric requires ground truth. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateCore(GroundedAnswer<>,String)` | Evaluates context coverage. |
| `EvaluateWithGroundTruth(GroundedAnswer<>,String)` | Evaluates coverage when ground truth is available. |
| `EvaluateWithoutGroundTruth(GroundedAnswer<>)` | Evaluates coverage without ground truth using heuristics. |

