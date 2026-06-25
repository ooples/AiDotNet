---
title: "CurriculumLearningOptions<T, TInput, TOutput>"
description: "Configuration options for Curriculum Learning through the AiDotNet facade."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for Curriculum Learning through the AiDotNet facade.

## For Beginners

Curriculum Learning trains models by presenting samples in order of difficulty,
starting with easy examples and gradually introducing harder ones. This often leads to faster
convergence and better final performance compared to random training order.

## How It Works

This options class is designed for use with `AiModelBuilder`.
It follows the AiDotNet facade pattern: users provide minimal configuration, and the library supplies
industry-standard defaults internally.

**Key Concepts:**

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `CompetenceBased` | Gets or sets competence-based learning options. |
| `CustomDifficultyEstimator` | Gets or sets a pre-built difficulty estimator. |
| `CustomScheduler` | Gets or sets a custom curriculum scheduler. |
| `Dataset` | Gets or sets the dataset on which to run the curriculum-learning training pass. |
| `DifficultyEstimator` | Gets or sets the difficulty estimator type that determines sample difficulty. |
| `DifficultyRecalculationFrequency` | Gets or sets how often to recalculate difficulties (in epochs). |
| `EarlyStopping` | Gets or sets early stopping options for curriculum learning. |
| `FinalDataFraction` | Gets or sets the final fraction of data to use (typically 1.0 for all samples). |
| `InitialDataFraction` | Gets or sets the initial fraction of data to use (easiest samples). |
| `NormalizeDifficulties` | Gets or sets whether to normalize difficulty scores to [0, 1]. |
| `NumPhases` | Gets or sets the number of curriculum phases. |
| `RandomSeed` | Gets or sets a random seed for reproducibility. |
| `RecalculateDifficulties` | Gets or sets whether to recalculate sample difficulties during training. |
| `ScheduleType` | Gets or sets the curriculum schedule type that controls progression from easy to hard samples. |
| `SelfPaced` | Gets or sets self-paced learning options. |
| `ShuffleWithinPhase` | Gets or sets whether to shuffle samples within each curriculum phase. |
| `TotalEpochs` | Gets or sets the total number of training epochs. |
| `UseDifficultyWeighting` | Gets or sets whether to weight sample contributions by difficulty. |
| `Verbosity` | Gets or sets the verbosity level for logging. |

