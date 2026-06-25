---
title: "ConfidenceBasedAdaptiveStrategy<T>"
description: "Adaptive distillation strategy that adjusts temperature based on student confidence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Adaptive distillation strategy that adjusts temperature based on student confidence.

## For Beginners

This strategy adapts temperature based on how confident
the student is in its predictions. When the student is confident (high max probability),
we use lower temperature (harder distillation). When uncertain (low max probability),
we use higher temperature (easier distillation with softer targets).

## How It Works

**Intuition:**

- **High Confidence** → Student understands this sample → Lower temp (sharpen targets)
- **Low Confidence** → Student struggles with this sample → Higher temp (soften targets)

**Example:**
Student predicts [0.95, 0.03, 0.02] → High confidence (0.95) → Low temperature
Student predicts [0.40, 0.35, 0.25] → Low confidence (0.40) → High temperature

**Best For:**

- General-purpose adaptive distillation
- When you want automatic difficulty adjustment
- Datasets with varying sample complexity

**Temperature Mapping:**
Confidence = max(probabilities)
Difficulty = 1 - Confidence
Temperature = MinTemp + Difficulty * (MaxTemp - MinTemp)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConfidenceBasedAdaptiveStrategy(Double,Double,Double,Double,Double)` | Initializes a new instance of the ConfidenceBasedAdaptiveStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAdaptiveTemperature(Vector<>,Vector<>)` | Computes adaptive temperature based on student confidence. |

