---
title: "CurriculumLearnerConfig<T>"
description: "Configuration for curriculum learning training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.CurriculumLearning`

Configuration for curriculum learning training.

## For Beginners

This class holds all the settings that control how
curriculum learning works. You can configure things like how many training phases
to use, when to stop early if the model isn't improving, and how to schedule
the progression from easy to hard samples.

## How It Works

**Key Configuration Areas:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CurriculumLearnerConfig` | Initializes a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets the batch size for training. |
| `DifficultyRecalculationFrequency` | Gets how often to recalculate difficulties (every N epochs). |
| `EarlyStoppingMinDelta` | Gets the minimum improvement required to reset early stopping counter. |
| `EarlyStoppingPatience` | Gets the number of epochs without improvement before early stopping. |
| `EpochsPerPhase` | Gets the number of epochs per phase. |
| `FinalDataFraction` | Gets the final data fraction (usually 1.0 to include all samples). |
| `InitialDataFraction` | Gets the initial data fraction (starting fraction of easiest samples). |
| `LearningRate` | Gets the learning rate. |
| `LogAction` | Gets the custom logging action. |
| `NormalizeDifficulties` | Gets whether to normalize difficulty scores to [0, 1]. |
| `NumPhases` | Gets the number of curriculum phases. |
| `RandomSeed` | Gets the random seed for reproducibility. |
| `RecalculateDifficulties` | Gets whether to recalculate sample difficulties during training. |
| `ScheduleType` | Gets the curriculum schedule type. |
| `ShuffleWithinPhase` | Gets whether to shuffle samples within each phase. |
| `TotalEpochs` | Gets the total number of training epochs. |
| `UseDifficultyWeighting` | Gets whether to weight sample contributions by difficulty. |
| `UseEarlyStopping` | Gets whether early stopping is enabled. |
| `Verbosity` | Gets the verbosity level for logging. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a copy of this configuration. |
| `CreateBuilder` | Creates a builder for fluent configuration. |

