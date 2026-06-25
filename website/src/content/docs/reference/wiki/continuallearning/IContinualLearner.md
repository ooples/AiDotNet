---
title: "IContinualLearner<T, TInput, TOutput>"
description: "Interface for continual learning trainers that can learn multiple tasks sequentially."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ContinualLearning.Interfaces`

Interface for continual learning trainers that can learn multiple tasks sequentially.

## For Beginners

Continual learning (also called lifelong learning) is the ability
to learn new tasks over time without forgetting previously learned knowledge. Traditional
neural networks suffer from "catastrophic forgetting" - when trained on new data, they
forget what they learned before.

## How It Works

**This interface provides methods to:**

- Learn new tasks while protecting old knowledge
- Evaluate performance on all learned tasks
- Save and load model state for resuming training

**Common Implementations:**

- **EWCTrainer:** Uses Elastic Weight Consolidation to protect important weights
- **LwFTrainer:** Uses Learning without Forgetting with knowledge distillation
- **GEMTrainer:** Uses Gradient Episodic Memory to constrain gradients

**References:**

- Parisi et al. "Continual Lifelong Learning with Neural Networks: A Review" (2019)
- De Lange et al. "A Continual Learning Survey" (2021)

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseModel` | Gets the underlying model being trained. |
| `Config` | Gets the configuration for the continual learner. |
| `IsTraining` | Gets whether the learner is currently training. |
| `MemoryUsageBytes` | Gets the current memory usage of the learner in bytes. |
| `TasksLearned` | Gets the number of tasks that have been learned. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeForgetting` | Computes the current forgetting metric for all tasks. |
| `EvaluateAllTasks` | Evaluates the model on all learned tasks. |
| `EvaluateTask(Int32,IDataset<,,>)` | Evaluates the model on a specific task. |
| `GetAllHistory` | Gets all training history. |
| `GetTaskHistory(Int32)` | Gets the training history for a specific task. |
| `LearnTask(IDataset<,,>)` | Learns a new task from the provided data. |
| `LearnTask(IDataset<,,>,IDataset<,,>,Nullable<Int32>)` | Learns a new task with a validation set for early stopping and monitoring. |
| `Load(String)` | Loads the learner state from a file. |
| `Reset` | Resets the learner to its initial state. |
| `Save(String)` | Saves the learner state to a file. |

## Events

| Event | Summary |
|:-----|:--------|
| `EpochCompleted` | Event raised when an epoch completes during training. |
| `TaskCompleted` | Event raised when a task finishes training. |
| `TaskStarted` | Event raised when a task starts training. |

