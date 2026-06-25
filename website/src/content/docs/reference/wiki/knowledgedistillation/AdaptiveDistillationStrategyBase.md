---
title: "AdaptiveDistillationStrategyBase<T>"
description: "Abstract base class for adaptive distillation strategies with performance tracking."
section: "API Reference"
---

`Base Classes` · `AiDotNet.KnowledgeDistillation.Strategies`

Abstract base class for adaptive distillation strategies with performance tracking.

## For Beginners

This base class provides common functionality for all adaptive
strategies, including performance tracking with exponential moving average and temperature
range management.

## How It Works

**For Implementers:** Derive from this class and implement
`Vector{` to define your specific adaptation logic.

**Shared Features:**

- Exponential moving average (EMA) for performance tracking
- Temperature range validation and enforcement
- Performance history management
- Helper methods for confidence, entropy, and accuracy calculations

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaptiveDistillationStrategyBase(Double,Double,Double,Double,Double)` | Initializes a new instance of the AdaptiveDistillationStrategyBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationRate` | Gets the adaptation rate for exponential moving average. |
| `MaxTemperature` | Gets the maximum temperature for adaptation. |
| `MinTemperature` | Gets the minimum temperature for adaptation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ArgMax(Vector<>)` | Finds the index of the maximum value in a vector. |
| `ClampTemperature(Double)` | Clamps a value to the temperature range [MinTemperature, MaxTemperature]. |
| `ComputeAdaptiveTemperature(Vector<>,Vector<>)` | Computes the adaptive temperature for a specific sample. |
| `ComputeEntropy(Vector<>)` | Computes the entropy of a probability distribution. |
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes gradient with adaptive temperature. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes distillation loss with adaptive temperature. |
| `ComputePerformance(Vector<>,Vector<>)` | Computes a performance metric for the student output. |
| `GetMaxConfidence(Vector<>)` | Gets the maximum confidence (highest probability) from a probability distribution. |
| `GetPerformance(Int32)` | Gets the current performance metric for a sample. |
| `IsCorrect(Vector<>,Vector<>)` | Checks if the student prediction is correct. |
| `UpdatePerformance(Int32,Vector<>,Vector<>)` | Updates the performance metric for a specific sample using exponential moving average. |

