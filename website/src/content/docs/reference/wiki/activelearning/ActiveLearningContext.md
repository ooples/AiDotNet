---
title: "ActiveLearningContext<T>"
description: "Context information for stopping criterion evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Interfaces`

Context information for stopping criterion evaluation.

## For Beginners

This class provides all the information a stopping criterion
needs to make its decision, including history of metrics and current state.

## Properties

| Property | Summary |
|:-----|:--------|
| `AccuracyHistory` | Gets or sets the history of training accuracy. |
| `CurrentIteration` | Gets or sets the current iteration number. |
| `CurrentPredictions` | Gets or sets the current predictions for stability checking. |
| `ElapsedTime` | Gets or sets the time elapsed since learning started. |
| `LossHistory` | Gets or sets the history of training loss. |
| `MaxBudget` | Gets or sets the maximum labeling budget. |
| `MaxTime` | Gets or sets the maximum allowed time for learning. |
| `PreviousPredictions` | Gets or sets the predictions from the previous iteration for stability checking. |
| `QueryScoreHistory` | Gets or sets the history of query informativeness scores. |
| `TotalLabeled` | Gets or sets the total number of labeled samples. |
| `UncertaintyHistory` | Gets or sets the history of average uncertainty scores. |
| `UnlabeledRemaining` | Gets or sets the number of unlabeled samples remaining. |
| `ValidationAccuracyHistory` | Gets or sets the history of validation accuracy. |

