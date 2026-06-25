---
title: "HybridDistillationStrategy<T>"
description: "Hybrid distillation strategy that combines multiple distillation strategies with configurable weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Hybrid distillation strategy that combines multiple distillation strategies with configurable weights.

## How It Works

**For Production Use:** This strategy allows you to combine multiple distillation approaches
(response-based, feature-based, attention-based, etc.) in a single training run. Each strategy
contributes to the total loss based on its configured weight.

**Example Use Case:**
For transformer distillation, combine:

- 40% Response-based (output matching)
- 30% Attention-based (attention pattern matching)
- 30% Feature-based (intermediate layer matching)

This gives you comprehensive knowledge transfer at multiple levels.

**Benefits:**

- Leverages multiple knowledge transfer mechanisms simultaneously
- Weights can be tuned based on validation performance
- More robust than single-strategy distillation
- Commonly used in SOTA models like TinyBERT, MobileBERT

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridDistillationStrategy(ValueTuple<IDistillationStrategy<>,Double>[],Double,Double)` | Initializes a new instance of the HybridDistillationStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes combined gradient from all strategies. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes combined loss from all strategies. |
| `GetStrategies` | Gets the individual strategies and their weights. |

