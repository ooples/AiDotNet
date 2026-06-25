---
title: "ICurriculumLearnerConfig<T>"
description: "Configuration interface for curriculum learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.CurriculumLearning.Interfaces`

Configuration interface for curriculum learning.

## For Beginners

This configuration controls how curriculum learning
behaves. You can set things like how many epochs to train, how to schedule
the introduction of harder samples, and when to stop training.

## How It Works

**Key Configuration Areas:**

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets the batch size for training. |
| `DifficultyRecalculationFrequency` | Gets how often to recalculate difficulties (in epochs). |
| `EarlyStoppingMinDelta` | Gets the minimum improvement required to reset patience. |
| `EarlyStoppingPatience` | Gets the patience for early stopping (epochs without improvement). |
| `EpochsPerPhase` | Gets the number of epochs per curriculum phase. |
| `FinalDataFraction` | Gets the final data fraction (at final phase). |
| `InitialDataFraction` | Gets the initial data fraction (at phase 0). |
| `LearningRate` | Gets the learning rate. |
| `LogAction` | Gets the custom logging action. |
| `NormalizeDifficulties` | Gets whether to normalize difficulty scores to [0, 1]. |
| `NumPhases` | Gets the number of curriculum phases. |
| `RandomSeed` | Gets the random seed for reproducibility. |
| `RecalculateDifficulties` | Gets whether to recalculate difficulty scores during training. |
| `ScheduleType` | Gets the type of curriculum schedule to use. |
| `ShuffleWithinPhase` | Gets whether to shuffle within curriculum phases. |
| `TotalEpochs` | Gets the total number of training epochs. |
| `UseDifficultyWeighting` | Gets whether to apply sample weighting based on difficulty. |
| `UseEarlyStopping` | Gets whether to use early stopping. |
| `Verbosity` | Gets the verbosity level for logging. |

