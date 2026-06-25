---
title: "MetaEvaluationResult<T>"
description: "Results from evaluating a meta-learner across multiple tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Results from evaluating a meta-learner across multiple tasks.

## For Beginners

Meta-learning evaluation tests how well your model can quickly learn new tasks.

This result tells you:

- **Average accuracy:** How well the model performs on new tasks after quick adaptation
- **Consistency:** How much performance varies across different tasks (standard deviation)
- **Confidence:** Statistical confidence intervals for the results
- **Per-task details:** Individual task results for deep analysis

For example, if you're doing 5-way 1-shot classification:

- You sample many test tasks (e.g., 100 tasks)
- For each task, the model sees 1 example per class and must classify new examples
- This result aggregates accuracy and loss across all those tasks
- Higher mean accuracy and lower standard deviation indicate better meta-learning

## How It Works

This class aggregates evaluation metrics from multiple tasks to assess meta-learning performance.
It uses the existing BasicStats infrastructure to provide comprehensive statistical analysis
of how well the meta-learner adapts to new tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaEvaluationResult(Vector<>,Vector<>,TimeSpan,Dictionary<String,>)` | Initializes a new instance with task results and calculates all statistics. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccuracyStats` | Gets comprehensive statistics for accuracy across all evaluated tasks. |
| `AdditionalMetrics` | Gets algorithm-specific metrics that don't fit standard categories. |
| `EvaluationTime` | Gets the total time taken for evaluation. |
| `LossStats` | Gets comprehensive statistics for loss across all evaluated tasks. |
| `NumTasks` | Gets the number of tasks used for evaluation. |
| `PerTaskAccuracies` | Gets the individual accuracy values for each evaluated task. |
| `PerTaskLosses` | Gets the individual loss values for each evaluated task. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateReport` | Generates a formatted summary report of the evaluation results. |
| `GetAccuracyConfidenceInterval` | Calculates the 95% confidence interval for mean accuracy. |
| `GetLossConfidenceInterval` | Calculates the 95% confidence interval for mean loss. |

