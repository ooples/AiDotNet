---
title: "IAdaptiveDistillationStrategy<T>"
description: "Interface for adaptive distillation strategies that adjust temperature based on student performance."
section: "API Reference"
---

`Interfaces` · `AiDotNet.KnowledgeDistillation.Strategies`

Interface for adaptive distillation strategies that adjust temperature based on student performance.

## For Beginners

Adaptive strategies dynamically adjust the temperature parameter
during training based on how well the student is learning. This allows for more flexible
knowledge transfer compared to fixed-temperature distillation.

## How It Works

**Key Concepts:**

- **Performance Tracking**: Monitor student learning progress
- **Temperature Adaptation**: Adjust temperature based on sample difficulty or student confidence
- **Per-Sample Adjustment**: Different temperatures for different training samples

**When to Use:**

- Training data has varying difficulty levels
- Student performance is uneven across samples
- You want automatic temperature tuning instead of manual selection

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationRate` | Gets the adaptation rate for performance tracking. |
| `MaxTemperature` | Gets the maximum temperature value used for adaptation. |
| `MinTemperature` | Gets the minimum temperature value used for adaptation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAdaptiveTemperature(Vector<>,Vector<>)` | Computes the adaptive temperature for a specific sample. |
| `GetPerformance(Int32)` | Gets the current performance metric for a specific sample. |
| `UpdatePerformance(Int32,Vector<>,Vector<>)` | Updates the performance metric for a specific training sample. |

