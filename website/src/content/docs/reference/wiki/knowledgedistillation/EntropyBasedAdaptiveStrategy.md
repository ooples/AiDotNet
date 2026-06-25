---
title: "EntropyBasedAdaptiveStrategy<T>"
description: "Adaptive distillation strategy that adjusts temperature based on prediction entropy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Adaptive distillation strategy that adjusts temperature based on prediction entropy.

## For Beginners

Entropy measures how uncertain or "spread out" a probability
distribution is. High entropy means the student is uncertain (probabilities are similar
across classes). Low entropy means the student is certain (one class has high probability).

## How It Works

**Entropy Examples:**

- **Low Entropy** [0.95, 0.03, 0.02]: Student is certain → Class 0 dominates
- **High Entropy** [0.35, 0.33, 0.32]: Student is uncertain → All classes similar

**Intuition:**

- **High Entropy** (uncertain) → Student struggling → Lower temp (focus learning)
- **Low Entropy** (certain) → Student confident → Higher temp (explore more)

**Why Lower Temp for High Entropy?**
When student is uncertain, we want to provide sharper (lower temp) targets to focus
learning on the most important features, rather than soft targets that might reinforce
uncertainty.

**Best For:**

- Detecting student uncertainty
- Calibrating overconfident students
- Datasets where uncertainty patterns are meaningful

**Entropy Range:**

- Minimum: 0.0 (completely certain, one class = 1.0)
- Maximum: 1.0 (normalized, completely uncertain, uniform distribution)
- Normalized by log(num_classes) to get [0, 1] range

**Temperature Mapping:**
High entropy → high difficulty → lower temperature (sharpen)
Low entropy → low difficulty → higher temperature (soften)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EntropyBasedAdaptiveStrategy(Double,Double,Double,Double,Double)` | Initializes a new instance of the EntropyBasedAdaptiveStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAdaptiveTemperature(Vector<>,Vector<>)` | Computes adaptive temperature based on prediction entropy. |
| `ComputePerformance(Vector<>,Vector<>)` | Computes performance based on entropy (inverse relationship). |

