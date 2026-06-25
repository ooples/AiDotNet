---
title: "TaskEvaluationResult<T>"
description: "Result from evaluating model performance on a single task."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Results`

Result from evaluating model performance on a single task.

## For Beginners

This class captures how well the model performs on a specific task
at a point in time. It's used to track whether the model remembers what it learned.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TaskEvaluationResult(Int32,,)` | Initializes a new instance of the `TaskEvaluationResult` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Accuracy` | Gets the accuracy on this task. |
| `AdditionalMetrics` | Gets additional metrics like F1-score, precision, recall. |
| `ConfidenceScores` | Gets the confidence scores for predictions, if available. |
| `ConfusionMatrix` | Gets the confusion matrix, if available. |
| `CorrectCount` | Gets the number of correct predictions. |
| `EvaluationTime` | Gets the evaluation time. |
| `Loss` | Gets the loss on this task. |
| `PerClassAccuracy` | Gets per-class accuracy breakdown, if available. |
| `SampleCount` | Gets the number of samples evaluated. |
| `TaskId` | Gets the task identifier (0-indexed). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a string representation of the evaluation result. |

