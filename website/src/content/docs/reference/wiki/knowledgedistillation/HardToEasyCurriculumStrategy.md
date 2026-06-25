---
title: "HardToEasyCurriculumStrategy<T>"
description: "Curriculum distillation strategy that progresses from hard to easy samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Curriculum distillation strategy that progresses from hard to easy samples.

## For Beginners

Hard-to-easy curriculum is the opposite of traditional
curriculum learning. It starts with challenging samples and gradually includes easier ones.
This approach is useful for fine-tuning already-trained models or when the student
has prior knowledge.

## How It Works

**How It Works:**

1. **Early Training** (progress 0.0-0.3):
- Include only hard samples (difficulty ≥ 0.7)
- Use low temperature (sharp targets, challenging)
2. **Mid Training** (progress 0.3-0.7):
- Include hard and medium samples (difficulty ≥ 0.3)
- Gradually increase temperature
3. **Late Training** (progress 0.7-1.0):
- Include all samples (even easy ones)
- Use high temperature (soft targets, exploratory)

**Temperature Progression:**
Starts at MinTemperature (e.g., 2.0) and linearly increases to MaxTemperature (e.g., 5.0)
as training progresses. This makes distillation progressively easier.

**Sample Filtering:**
At progress P, only include samples with difficulty ≥ (1 - P).
Example: At 50% progress, only samples with difficulty ≥ 0.5 are included.

**Real-World Analogy:**
Training an advanced student: Start with challenging problems to identify gaps in knowledge,
then fill in easier concepts they might have missed. Like a PhD student reviewing
undergraduate material to strengthen foundations.

**When to Use Hard-to-Easy:**

- **Fine-tuning**: Student already has base knowledge
- **Transfer Learning**: Adapting pre-trained model to new domain
- **Anti-forgetting**: Prevent model from forgetting hard concepts
- **Expert Refinement**: Polish already-good student model
- **Debugging**: Identify which hard samples student struggles with

**Advantages:**

- Forces student to tackle challenges early
- Identifies weaknesses quickly
- Can improve performance on difficult edge cases
- Prevents overfitting to easy samples

**Disadvantages:**

- Can be unstable if student has no prior knowledge
- May not converge well from random initialization
- Harder to tune than easy-to-hard

**References:**

- Krueger & Dayan (2009). Flexible shaping: How learning in small steps helps.
- Kumar et al. (2010). Self-paced curriculum learning (discusses anti-curriculum).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HardToEasyCurriculumStrategy(Double,Double,Double,Double,Int32,Dictionary<Int32,Double>)` | Initializes a new instance of the HardToEasyCurriculumStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCurriculumTemperature` | Computes curriculum temperature that increases over time (hard to easy). |
| `ShouldIncludeSample(Int32)` | Determines if a sample should be included based on curriculum progress (inverted). |

