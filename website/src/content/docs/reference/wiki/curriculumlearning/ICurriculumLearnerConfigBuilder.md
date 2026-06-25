---
title: "ICurriculumLearnerConfigBuilder<T>"
description: "Builder for curriculum learner configuration."
section: "API Reference"
---

`Interfaces` · `AiDotNet.CurriculumLearning.Interfaces`

Builder for curriculum learner configuration.

## Methods

| Method | Summary |
|:-----|:--------|
| `Build` | Builds the configuration. |
| `WithBatchSize(Int32)` | Sets the batch size. |
| `WithDifficultyRecalculation(Int32)` | Enables difficulty recalculation during training. |
| `WithEarlyStopping(Int32,)` | Configures early stopping. |
| `WithInitialDataFraction()` | Sets the initial data fraction. |
| `WithLearningRate()` | Sets the learning rate. |
| `WithLogAction(Action<String>)` | Sets a custom logging action. |
| `WithNumPhases(Int32)` | Sets the number of curriculum phases. |
| `WithRandomSeed(Int32)` | Sets the random seed. |
| `WithScheduleType(CurriculumScheduleType)` | Sets the curriculum schedule type. |
| `WithTotalEpochs(Int32)` | Sets the total number of training epochs. |
| `WithVerbosity(CurriculumVerbosity)` | Sets the verbosity level. |

