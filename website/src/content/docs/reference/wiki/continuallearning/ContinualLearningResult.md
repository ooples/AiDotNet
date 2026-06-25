---
title: "ContinualLearningResult<T>"
description: "Result from training on a single task in continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Results`

Result from training on a single task in continual learning.

## For Beginners

This class captures everything that happened during training
on one task - how well the model learned, how long it took, and whether it forgot
previous knowledge.

## How It Works

**Key Metrics Explained:**

- **Training Loss:** How far predictions were from targets (lower is better)
- **Training Accuracy:** Percentage of correct predictions on training data
- **Average Previous Task Accuracy:** How well the model still performs on old tasks
- **Forgetting:** How much accuracy was lost on previous tasks (lower is better)

**Reference:** Parisi et al. "Continual Lifelong Learning with Neural Networks: A Review" (2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContinualLearningResult(Int32,,,,TimeSpan,Vector<>,Vector<>)` | Initializes a new instance of the `ContinualLearningResult` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AveragePreviousTaskAccuracy` | Gets the average accuracy on all previously learned tasks after training on this task. |
| `EffectiveLearningRate` | Gets the effective learning rate used (may vary with schedulers). |
| `Forgetting` | Gets the amount of forgetting on previous tasks (accuracy drop). |
| `ForwardTransfer` | Gets the forward transfer metric. |
| `GradientUpdates` | Gets the number of gradient updates performed. |
| `LossHistory` | Gets the loss history across training epochs. |
| `PeakMemoryBytes` | Gets the peak memory usage during training in bytes. |
| `RegularizationLossHistory` | Gets the regularization loss history (e.g., EWC penalty term). |
| `SampleCount` | Gets the number of samples used for training. |
| `StrategyMetrics` | Gets additional strategy-specific metrics. |
| `TaskId` | Gets the task identifier (0-indexed). |
| `TrainingAccuracy` | Gets the final training accuracy on this task. |
| `TrainingLoss` | Gets the final training loss on this task. |
| `TrainingTime` | Gets the time taken to train on this task. |
| `ValidationAccuracy` | Gets the validation accuracy on this task, if validation was performed. |
| `ValidationLoss` | Gets the validation loss on this task, if validation was performed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a string representation of the training result. |

