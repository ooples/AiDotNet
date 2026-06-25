---
title: "RelationAggregationMethod"
description: "Methods for aggregating multiple relation scores in Relation Networks."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Methods for aggregating multiple relation scores in Relation Networks.

## For Beginners

In few-shot learning, each class has several example
images. When classifying a new query image, we compare it to ALL examples of each
class. This enum controls how those multiple similarity scores are combined into
a single score for each class.

For example, if we have 5 dog examples and compare a query to each:

- Mean: Average all 5 scores
- Max: Take the highest score (most similar dog example)
- Attention: Weight scores by relevance
- LearnedWeighting: Let the network learn optimal weights

## How It Works

When there are multiple support examples per class, we need a way to combine
the relation scores from comparing a query with each support example.

## Fields

| Field | Summary |
|:-----|:--------|
| `Attention` | Use attention-weighted average. |
| `LearnedWeighting` | Use learned weighting. |
| `Max` | Use maximum score. |
| `Mean` | Compute mean of all scores. |

