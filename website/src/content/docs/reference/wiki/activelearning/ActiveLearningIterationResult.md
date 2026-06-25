---
title: "ActiveLearningIterationResult<T>"
description: "Result from a single active learning iteration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Results`

Result from a single active learning iteration.

## For Beginners

Each iteration of active learning involves selecting samples,
getting labels, and retraining the model. This class captures what happened in one iteration.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ActiveLearningIterationResult(INumericOperations<>)` | Initializes a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AveragePoolUncertainty` | Gets or sets the average uncertainty of the unlabeled pool. |
| `AverageQueryScore` | Gets or sets the average informativeness score of queried samples. |
| `HasValidation` | Gets or sets whether validation data was available for this iteration. |
| `IterationNumber` | Gets or sets the iteration number (0-indexed). |
| `IterationTime` | Gets or sets the time spent on this iteration. |
| `MaxQueryScore` | Gets or sets the maximum informativeness score among queried samples. |
| `QueriedIndices` | Gets or sets the indices of samples that were queried. |
| `QueryScores` | Gets or sets the informativeness scores of queried samples. |
| `SamplesQueried` | Gets or sets the number of samples queried in this iteration. |
| `SelectionTime` | Gets or sets the time spent selecting samples. |
| `TotalLabeledSamples` | Gets or sets the total number of labeled samples after this iteration. |
| `TrainingAccuracy` | Gets or sets the training accuracy after this iteration. |
| `TrainingLoss` | Gets or sets the training loss after this iteration. |
| `TrainingTime` | Gets or sets the time spent training the model. |
| `UnlabeledRemaining` | Gets or sets the number of unlabeled samples remaining. |
| `ValidationAccuracy` | Gets or sets the validation accuracy after this iteration (if available). |
| `ValidationLoss` | Gets or sets the validation loss after this iteration (if available). |

