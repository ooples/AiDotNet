---
title: "ContinualLearnerBase<T, TInput, TOutput>"
description: "Base class for continual learning trainers that provides common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ContinualLearning.Trainers`

Base class for continual learning trainers that provides common functionality.

## How It Works

**For Algorithm Implementers:** To create a new continual learning algorithm:

**Common Patterns:**

**Reference:** De Lange et al. "A Continual Learning Survey: Defying Forgetting" (2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContinualLearnerBase(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>)` | Initializes a new continual learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseModel` |  |
| `Config` |  |
| `Engine` | Provides hardware-accelerated tensor/vector operations. |
| `IsTraining` |  |
| `MemoryUsageBytes` |  |
| `TasksLearned` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAveragePreviousTaskAccuracy` | Computes the average accuracy on all previously learned tasks. |
| `ComputeCurrentForgetting` | Computes the current forgetting averaged across all tasks. |
| `ComputeForgetting` |  |
| `EvaluateAllTasks` |  |
| `EvaluateOnDataset(IDataset<,,>)` | Evaluates the model on a dataset and returns simple metrics. |
| `EvaluateTask(Int32,IDataset<,,>)` |  |
| `GetAllHistory` |  |
| `GetClassLabel()` | Gets the class label from an output. |
| `GetTaskHistory(Int32)` |  |
| `IsPredictionCorrect(,)` | Determines if a prediction is correct. |
| `LearnTask(IDataset<,,>)` |  |
| `LearnTask(IDataset<,,>,IDataset<,,>,Nullable<Int32>)` |  |
| `Load(String)` |  |
| `OnEpochCompleted(Int32,Int32,Int32,,)` | Raises the EpochCompleted event. |
| `OnTaskCompleted(Int32,Int32,ContinualLearningResult<>)` | Raises the TaskCompleted event. |
| `OnTaskStarted(Int32,Int32)` | Raises the TaskStarted event. |
| `Reset` |  |
| `Save(String)` |  |
| `TrainOnTask(IDataset<,,>,IDataset<,,>,Int32)` | Performs the actual training on a task. |
| `ValidatePathWithinDirectory(String,String)` | Validates that a path is within the expected directory (security measure). |

## Fields

| Field | Summary |
|:-----|:--------|
| `Configuration` | Configuration for the continual learner. |
| `LossFunction` | The loss function used for training. |
| `MemoryBuffer` | Memory buffer for experience replay. |
| `Model` | The underlying model being trained. |
| `NumOps` | Numeric operations for generic type T. |
| `Strategy` | Strategy for preventing catastrophic forgetting. |
| `_initialAccuracies` | Initial accuracy on each task before it was learned (for forward transfer). |
| `_isTraining` | Whether the learner is currently training. |
| `_lock` | Synchronization lock for thread safety. |
| `_peakAccuracies` | Accuracy on each task right after it was learned (for backward transfer). |
| `_taskTestSets` | Test sets for each learned task (for evaluation). |
| `_tasksLearned` | Number of tasks successfully learned. |
| `_trainingHistory` | Training history for all tasks. |

## Events

| Event | Summary |
|:-----|:--------|
| `EpochCompleted` |  |
| `TaskCompleted` |  |
| `TaskStarted` |  |

