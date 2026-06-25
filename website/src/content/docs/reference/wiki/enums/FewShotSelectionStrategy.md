---
title: "FewShotSelectionStrategy"
description: "Represents strategies for selecting few-shot examples in prompt templates."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents strategies for selecting few-shot examples in prompt templates.

## For Beginners

Few-shot selection strategies determine which examples to show the language model.

Think of it like choosing which practice problems to show a student:

- You could show random problems
- You could show problems similar to the current one
- You could show problems that cover diverse scenarios
- You could show the most helpful examples

The right strategy depends on what you're trying to teach and what works best.
Different strategies can significantly impact the model's performance.

## Fields

| Field | Summary |
|:-----|:--------|
| `ClusterBased` | Select examples using a clustering approach to ensure broad coverage. |
| `Diversity` | Select diverse examples to maximize coverage of different patterns. |
| `Fixed` | Select examples in a fixed, predetermined order. |
| `MaximumMarginalRelevance` | Select examples based on maximum marginal relevance (balance between relevance and diversity). |
| `Random` | Select examples randomly from the available pool. |
| `SemanticSimilarity` | Select examples most semantically similar to the current input. |

