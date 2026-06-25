---
title: "ContinualEvaluationResult<T>"
description: "Comprehensive evaluation result across all learned tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Results`

Comprehensive evaluation result across all learned tasks.

## For Beginners

This class provides a complete picture of how well the model
performs across all tasks it has learned. It includes key metrics for measuring
continual learning effectiveness.

## How It Works

**Key Metrics Explained:**

- **Average Accuracy:** Mean accuracy across all tasks (higher is better)
- **Backward Transfer:** How learning new tasks affects old task performance (positive = improvement, negative = forgetting)
- **Forward Transfer:** How old knowledge helps with new tasks (positive = positive transfer)
- **Forgetting:** Maximum accuracy drop on any previous task (lower is better)

**Reference:** Lopez-Paz and Ranzato "Gradient Episodic Memory for Continual Learning" (2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContinualEvaluationResult(Vector<>,Vector<>,,,,,TimeSpan)` | Initializes a new instance of the `ContinualEvaluationResult` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccuracyMatrix` | Gets the accuracy matrix R where R[i,j] is accuracy on task j after learning task i. |
| `AverageAccuracy` | Gets the average accuracy across all tasks. |
| `AverageLoss` | Gets the average loss across all tasks. |
| `BackwardTransfer` | Gets the backward transfer metric. |
| `EvaluationTime` | Gets the time taken for evaluation. |
| `ForwardTransfer` | Gets the forward transfer metric. |
| `Intransigence` | Gets the intransigence metric (inability to learn new tasks). |
| `LearningEfficiency` | Gets the learning curve efficiency (area under the learning curve). |
| `MaxForgetting` | Gets the maximum forgetting observed on any task. |
| `PerTaskResults` | Gets the per-task evaluation results. |
| `StabilityPlasticityRatio` | Gets the stability-plasticity ratio. |
| `TaskAccuracies` | Gets the accuracy on each task. |
| `TaskCount` | Gets the total number of tasks evaluated. |
| `TaskLosses` | Gets the loss on each task. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateReport` | Generates a detailed report of the evaluation results. |
| `ToString` | Returns a string representation of the evaluation result. |

