---
title: "AggregationMode"
description: "Specifies how multiple teacher outputs are combined into a single supervision signal."
section: "API Reference"
---

`Enums` · `AiDotNet.KnowledgeDistillation.Teachers`

Specifies how multiple teacher outputs are combined into a single supervision signal.

## For Beginners

If you have multiple “experts” (teachers), this chooses how to combine their answers.
You can average their confidence scores, or you can let them vote on the final answer.

## How It Works

When knowledge distillation uses more than one teacher model, their predictions must be aggregated
before being used to train the student. This enum selects the aggregation strategy.

## Fields

| Field | Summary |
|:-----|:--------|
| `Average` | Averages the teachers' predictions. |
| `Voting` | Uses majority voting to select the final teacher prediction. |

