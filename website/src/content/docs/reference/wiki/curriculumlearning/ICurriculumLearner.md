---
title: "ICurriculumLearner<T, TInput, TOutput>"
description: "Interface for curriculum learning trainers that train models using a structured curriculum."
section: "API Reference"
---

`Interfaces` · `AiDotNet.CurriculumLearning.Interfaces`

Interface for curriculum learning trainers that train models using a structured curriculum.

## For Beginners

Curriculum learning is inspired by how humans learn - starting
with easy examples and gradually progressing to harder ones. Just like a student learns
basic arithmetic before calculus, a model trained with curriculum learning sees simple
examples first, then progressively harder ones.

## How It Works

**Why Curriculum Learning?**

**Key Components:**

**References:**

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseModel` | Gets the underlying model being trained. |
| `Config` | Gets the configuration for the curriculum learner. |
| `CurrentEpoch` | Gets the current epoch number. |
| `CurrentPhase` | Gets the current training phase (0-1, where 1 means all samples are available). |
| `DifficultyEstimator` | Gets the difficulty estimator used to rank samples. |
| `IsTraining` | Gets whether training is currently in progress. |
| `Scheduler` | Gets the curriculum scheduler that controls training progression. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdvancePhase` | Advances the curriculum to the next phase. |
| `EstimateDifficulties(IDataset<,,>)` | Estimates difficulty scores for all samples in a dataset. |
| `GetCurrentCurriculumIndices(Vector<>)` | Gets the indices of samples available at the current curriculum phase. |
| `GetPhaseHistory` | Gets the training history. |
| `Load(String)` | Loads the curriculum learner state. |
| `ResetCurriculum` | Resets the curriculum to the initial phase. |
| `Save(String)` | Saves the curriculum learner state. |
| `Train(IDataset<,,>)` | Trains the model using curriculum learning. |
| `Train(IDataset<,,>,IDataset<,,>)` | Trains the model with a validation set for monitoring. |
| `TrainWithDifficulty(IDataset<,,>,Vector<>)` | Trains with pre-computed difficulty scores. |

## Events

| Event | Summary |
|:-----|:--------|
| `PhaseCompleted` | Event raised when a curriculum phase completes. |
| `PhaseStarted` | Event raised when a curriculum phase starts. |
| `TrainingCompleted` | Event raised when training completes. |

