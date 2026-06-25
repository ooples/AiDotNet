---
title: "CurriculumLearnerConfigBuilder<T>"
description: "Fluent builder for curriculum learning configuration."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.CurriculumLearning`

Fluent builder for curriculum learning configuration.

## For Beginners

This builder lets you configure curriculum learning
using a fluent, readable style. Each method returns the builder so you can
chain multiple settings together.

## How It Works

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CurriculumLearnerConfigBuilder` | Initializes a new instance with default values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Build` |  |
| `Reset` | Resets the builder to default values. |
| `WithBatchSize(Int32)` |  |
| `WithDifficultyRecalculation(Boolean,Int32)` | Enables or disables difficulty recalculation during training. |
| `WithDifficultyRecalculation(Int32)` |  |
| `WithDifficultyWeighting(Boolean)` |  |
| `WithEarlyStopping(Int32,)` |  |
| `WithFinalDataFraction()` |  |
| `WithInitialDataFraction()` |  |
| `WithLearningRate()` |  |
| `WithLogAction(Action<String>)` |  |
| `WithNormalizeDifficulties(Boolean)` |  |
| `WithNumPhases(Int32)` |  |
| `WithRandomSeed(Int32)` |  |
| `WithScheduleType(CurriculumScheduleType)` |  |
| `WithShuffling(Boolean)` |  |
| `WithTotalEpochs(Int32)` |  |
| `WithVerbosity(CurriculumVerbosity)` |  |
| `WithoutEarlyStopping` |  |

