---
title: "EasyToHardCurriculumStrategy<T>"
description: "Curriculum distillation strategy that progresses from easy to hard samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Curriculum distillation strategy that progresses from easy to hard samples.

## For Beginners

Easy-to-hard curriculum learning mimics how humans learn best:
start with simple concepts and gradually introduce more complex ones. This strategy
filters training samples and adjusts temperature based on difficulty and training progress.

## How It Works

**How It Works:**

1. **Early Training** (progress 0.0-0.3):
- Include only easy samples (difficulty ≤ 0.3)
- Use high temperature (soft targets, gentle learning)
2. **Mid Training** (progress 0.3-0.7):
- Include easy and medium samples (difficulty ≤ 0.7)
- Gradually decrease temperature
3. **Late Training** (progress 0.7-1.0):
- Include all samples (even hard ones)
- Use low temperature (sharp targets, challenging)

**Temperature Progression:**
Starts at MaxTemperature (e.g., 5.0) and linearly decreases to MinTemperature (e.g., 2.0)
as training progresses. This makes distillation progressively more challenging.

**Sample Filtering:**
At progress P, only include samples with difficulty ≤ P.
Example: At 50% progress, only samples with difficulty ≤ 0.5 are included.

**Real-World Analogy:**
Learning mathematics: Start with addition (easy), then multiplication (medium),
then algebra (hard). Don't try to teach calculus to someone who hasn't learned addition!

**Best For:**

- Training from scratch
- Datasets with clear difficulty levels
- Preventing student from being overwhelmed early
- Improving convergence speed and final performance

**References:**

- Bengio et al. (2009). Curriculum Learning. ICML.
- Kumar et al. (2010). Self-paced Learning for Latent Variable Models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EasyToHardCurriculumStrategy(Double,Double,Double,Double,Int32,Dictionary<Int32,Double>)` | Initializes a new instance of the EasyToHardCurriculumStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCurriculumTemperature` | Computes curriculum temperature that decreases over time (easy to hard). |
| `ShouldIncludeSample(Int32)` | Determines if a sample should be included based on curriculum progress. |

