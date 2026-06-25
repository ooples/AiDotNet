---
title: "AccuracyBasedAdaptiveStrategy<T>"
description: "Adaptive distillation strategy that adjusts temperature based on student accuracy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Adaptive distillation strategy that adjusts temperature based on student accuracy.

## For Beginners

This strategy tracks whether the student is making correct
predictions and adjusts temperature accordingly. When the student is correct, we use
lower temperature (reinforce learning). When incorrect, we use higher temperature
(provide softer, more exploratory targets).

## How It Works

**Intuition:**

- **Correct Prediction** → Student learned this well → Lower temp (reinforce)
- **Incorrect Prediction** → Student struggling → Higher temp (help learn)

**Example:**
True label: [0, 1, 0] (class 1)
Student predicts: [0.1, 0.8, 0.1] → Correct! → Low temperature
Student predicts: [0.6, 0.3, 0.1] → Wrong! → High temperature

**Best For:**

- Supervised learning with labeled data
- When you want to focus more on difficult samples
- Tracking which samples student struggles with

**Requirements:**
Requires true labels to be provided in ComputeLoss/ComputeGradient calls.
Without labels, falls back to confidence-based adaptation.

**Performance Tracking:**
Uses exponential moving average of correctness:

- 1.0 = consistently correct
- 0.0 = consistently incorrect

Temperature inversely proportional to performance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AccuracyBasedAdaptiveStrategy(Double,Double,Double,Double,Double)` | Initializes a new instance of the AccuracyBasedAdaptiveStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAdaptiveTemperature(Vector<>,Vector<>)` | Computes adaptive temperature based on student accuracy. |
| `ComputePerformance(Vector<>,Vector<>)` | Computes performance based on prediction correctness. |

