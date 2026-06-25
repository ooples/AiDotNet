---
title: "CurriculumLearner<T, TInput, TOutput>"
description: "Main orchestrator for curriculum learning that coordinates difficulty estimation, scheduling, and model training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning`

Main orchestrator for curriculum learning that coordinates difficulty estimation,
scheduling, and model training.

## For Beginners

Curriculum learning is a training strategy that presents
samples from easy to hard, mimicking how humans learn. This class coordinates the
entire process:

## How It Works

**Key Components:**

**Example Usage:**

**References:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CurriculumLearner(IFullModel<,,>,ICurriculumLearnerConfig<>,IDifficultyEstimator<,,>,ICurriculumScheduler<>)` | Initializes a new instance of the `CurriculumLearner` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseModel` |  |
| `Config` |  |
| `CurrentEpoch` |  |
| `CurrentPhase` |  |
| `DifficultyEstimator` |  |
| `IsTraining` |  |
| `Scheduler` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdvancePhase` |  |
| `ArgMax(Vector<>)` | Returns the index of the maximum element in a vector. |
| `CheckEarlyStopping()` | Checks if early stopping should be triggered. |
| `CheckImprovement()` | Checks if the current loss represents an improvement. |
| `CompletePreviousPhase(CurriculumLearningResult<>,Int32,Int32,Int32,List<>,List<>)` | Completes a training phase and records results. |
| `ComputeLoss(,)` | Computes the loss for a single sample. |
| `CreateSchedulerFromConfig(ICurriculumLearnerConfig<>)` | Creates a scheduler based on configuration. |
| `EstimateDifficulties(IDataset<,,>)` |  |
| `Evaluate(IDataset<,,>)` | Evaluates the model on a dataset. |
| `FinalizeResult(CurriculumLearningResult<>,IDataset<,,>)` | Finalizes the training result with summary statistics. |
| `GetCurrentCurriculumIndices(Vector<>)` |  |
| `GetPhaseHistory` |  |
| `IsCorrectPrediction(,)` | Determines if a prediction is correct by comparing with expected output. |
| `Load(String)` |  |
| `LogMessage(String)` | Logs a message using the configured logging action or Console.WriteLine as fallback. |
| `LogPhaseCompleted(CurriculumPhaseResult<>)` | Logs a phase completion message based on verbosity. |
| `LogPhaseStart(Int32)` | Logs a phase start message based on verbosity. |
| `NormalizeDifficulties(Vector<>)` | Normalizes difficulty scores to [0, 1] range. |
| `RaisePhaseCompleted(CurriculumPhaseResult<>)` | Raises the PhaseCompleted event. |
| `RaisePhaseStarted(Int32,)` | Raises the PhaseStarted event. |
| `RaiseTrainingCompleted(CurriculumLearningResult<>)` | Raises the TrainingCompleted event. |
| `ResetCurriculum` |  |
| `Save(String)` |  |
| `ShuffleArray(Int32[])` | Shuffles an array in place using Fisher-Yates algorithm. |
| `Train(IDataset<,,>)` |  |
| `Train(IDataset<,,>,IDataset<,,>)` |  |
| `TrainEpoch(IDataset<,,>,Int32)` | Trains the model for one epoch on the provided data. |
| `TrainWithDifficulty(IDataset<,,>,Vector<>)` |  |
| `TrainWithDifficulty(IDataset<,,>,Vector<>,IDataset<,,>)` | Trains the model using curriculum learning with pre-computed difficulty scores and optional validation. |

## Events

| Event | Summary |
|:-----|:--------|
| `PhaseCompleted` |  |
| `PhaseStarted` |  |
| `TrainingCompleted` |  |

