---
title: "CurriculumDistillationStrategyBase<T>"
description: "Abstract base class for curriculum distillation strategies with progressive difficulty adjustment."
section: "API Reference"
---

`Base Classes` · `AiDotNet.KnowledgeDistillation.Strategies`

Abstract base class for curriculum distillation strategies with progressive difficulty adjustment.

## For Beginners

This base class provides common functionality for curriculum learning,
including progress tracking, sample difficulty management, and temperature progression.

## How It Works

**For Implementers:** Derive from this class and implement
`ComputeCurriculumTemperature` and `Int32)` to define
your specific curriculum progression logic.

**Shared Features:**

- Curriculum progress tracking (0.0 to 1.0)
- Sample difficulty scoring and management
- Temperature range validation
- Step/epoch-based progression

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CurriculumDistillationStrategyBase(Double,Double,Double,Double,Int32,Dictionary<Int32,Double>)` | Initializes a new instance of the CurriculumDistillationStrategyBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurriculumProgress` | Gets the current curriculum progress (0.0 to 1.0). |
| `MaxTemperature` | Gets the maximum temperature for the curriculum. |
| `MinTemperature` | Gets the minimum temperature for the curriculum. |
| `TotalSteps` | Gets the total number of steps in the curriculum. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClampTemperature(Double)` | Clamps a value to the temperature range [MinTemperature, MaxTemperature]. |
| `ComputeCurriculumTemperature` | Computes the curriculum-adjusted temperature based on current progress. |
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes gradient with curriculum-adjusted temperature. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes distillation loss with curriculum-adjusted temperature. |
| `GetSampleDifficulty(Int32)` | Gets the difficulty score for a sample, if set. |
| `SetSampleDifficulty(Int32,Double)` | Sets the difficulty score for a specific sample. |
| `ShouldIncludeSample(Int32)` | Determines if a sample should be included in current curriculum stage. |
| `UpdateProgress(Int32)` | Updates the current curriculum progress. |

