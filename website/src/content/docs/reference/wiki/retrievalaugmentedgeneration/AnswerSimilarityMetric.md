---
title: "AnswerSimilarityMetric<T>"
description: "Evaluates the similarity between the generated answer and ground truth."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Evaluation`

Evaluates the similarity between the generated answer and ground truth.

## For Beginners

This checks how close the AI's answer is to the correct answer.

Think of it like grading an exam:

- You have the answer key (ground truth)
- The student's answer (generated answer)
- This metric gives partial credit based on how much overlaps

For example:

- Ground truth: "Photosynthesis converts sunlight into energy"
- Generated: "Photosynthesis converts sunlight into chemical energy"
- Score: ~0.85 (most words match)

Scoring:

- 1.0: Perfect match
- 0.5-0.8: Partially correct
- 0.0-0.3: Mostly incorrect

Use cases:

- Benchmarking your RAG system against test datasets
- A/B testing different configurations
- Regression testing (ensure changes don't hurt quality)

Note: This uses simple word overlap. Production systems should use semantic
similarity with embeddings or BERTScore for better accuracy.

## How It Works

This metric measures how similar the generated answer is to a known correct answer (ground truth).
It uses Jaccard similarity (word overlap) to compare the two texts. This is useful for benchmarking
and regression testing when you have reference answers.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the description of what this metric measures. |
| `Name` | Gets the name of this metric. |
| `RequiresGroundTruth` | Gets a value indicating whether this metric requires ground truth. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateCore(GroundedAnswer<>,String)` | Evaluates similarity using Jaccard similarity. |

