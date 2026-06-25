---
title: "ICurriculumDistillationStrategy<T>"
description: "Interface for curriculum distillation strategies that progressively adjust training difficulty."
section: "API Reference"
---

`Interfaces` · `AiDotNet.KnowledgeDistillation.Strategies`

Interface for curriculum distillation strategies that progressively adjust training difficulty.

## For Beginners

Curriculum learning is inspired by how humans learn - starting
with easy concepts and gradually increasing difficulty. This interface defines strategies
that control this progression during knowledge distillation.

## How It Works

**Key Concepts:**

- **Progressive Difficulty**: Training difficulty increases (or decreases) over time
- **Sample Filtering**: Only include samples appropriate for current curriculum stage
- **Temperature Progression**: Temperature adjusts based on training progress

**Common Curriculum Strategies:**

- **Easy-to-Hard**: Start with simple samples, gradually add harder ones
- **Hard-to-Easy**: Start with challenging samples, then easier ones (for fine-tuning)
- **Paced Learning**: Combine difficulty-based and time-based progression

**When to Use:**

- Training data has clear difficulty levels
- Student model benefits from structured learning progression
- You want to prevent overwhelming the student early in training

## Properties

| Property | Summary |
|:-----|:--------|
| `CurriculumProgress` | Gets the current curriculum progress as a ratio (0.0 to 1.0). |
| `MaxTemperature` | Gets the maximum temperature for the curriculum range. |
| `MinTemperature` | Gets the minimum temperature for the curriculum range. |
| `TotalSteps` | Gets the total number of steps/epochs in the curriculum. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCurriculumTemperature` | Computes the curriculum-adjusted temperature based on current progress. |
| `GetSampleDifficulty(Int32)` | Gets the difficulty score for a specific sample, if set. |
| `SetSampleDifficulty(Int32,Double)` | Sets the difficulty score for a specific training sample. |
| `ShouldIncludeSample(Int32)` | Determines if a sample should be included in the current curriculum stage. |
| `UpdateProgress(Int32)` | Updates the current curriculum progress. |

