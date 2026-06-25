---
title: "FaithfulnessMetric<T>"
description: "Evaluates whether the generated answer is faithful to the source documents."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Evaluation`

Evaluates whether the generated answer is faithful to the source documents.

## For Beginners

This checks if the AI made stuff up or stuck to the sources.

Think of it like plagiarism checking in reverse:

- High score: The answer only says things found in the source documents
- Low score: The answer includes information not in the sources (hallucination)

For example:

- Sources say: "Photosynthesis produces oxygen"
- Faithful answer: "Photosynthesis produces oxygen" ✓ (score: 1.0)
- Unfaithful answer: "Photosynthesis produces oxygen and nitrogen" ✗ (score: 0.5)

Why this matters:

- Prevents the AI from making up facts
- Ensures answers are verifiable
- Builds user trust

Note: This is a simplified metric. Production systems should use more sophisticated
techniques like NLI (Natural Language Inference) models.

## How It Works

Faithfulness measures how well the generated answer adheres to the information in the
retrieved source documents. A faithful answer doesn't hallucinate or add information
not present in the sources. This metric checks for unsupported claims by analyzing
word overlap between the answer and source documents.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the description of what this metric measures. |
| `Name` | Gets the name of this metric. |
| `RequiresGroundTruth` | Gets a value indicating whether this metric requires ground truth. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateCore(GroundedAnswer<>,String)` | Evaluates faithfulness by measuring overlap between answer and sources. |

