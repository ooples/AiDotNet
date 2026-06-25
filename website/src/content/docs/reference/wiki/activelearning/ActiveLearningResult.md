---
title: "ActiveLearningResult<T>"
description: "Final result from the complete active learning process."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Results`

Final result from the complete active learning process.

## For Beginners

This class summarizes the entire active learning process,
including the learning curve, final performance, and efficiency metrics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ActiveLearningResult(INumericOperations<>)` | Initializes a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AreaUnderLearningCurve` | Gets or sets the Area Under the Learning Curve (AULC). |
| `BatchStrategyName` | Gets or sets the batch strategy used (if any). |
| `BudgetUsed` | Gets or sets the labeling budget used as a fraction of maximum. |
| `FinalTestAccuracy` | Gets or sets the final test accuracy. |
| `FinalTestLoss` | Gets or sets the final test loss. |
| `FinalTrainingAccuracy` | Gets or sets the final training accuracy. |
| `FinalTrainingLoss` | Gets or sets the final training loss. |
| `FinalValidationAccuracy` | Gets or sets the final validation accuracy. |
| `FinalValidationLoss` | Gets or sets the final validation loss. |
| `HasAULC` | Gets or sets whether AULC was calculated. |
| `HasQueryEfficiencyRatio` | Gets or sets whether query efficiency ratio was calculated. |
| `HasSampleEfficiency` | Gets or sets whether sample efficiency was calculated. |
| `HasTargetAccuracy` | Gets or sets whether target accuracy was set. |
| `HasTest` | Gets or sets whether test data was provided. |
| `HasValidation` | Gets or sets whether validation data was available. |
| `InitialLabeledSamples` | Gets or sets the initial number of labeled samples. |
| `IterationResults` | Gets or sets the results from each iteration. |
| `LearningCurve` | Gets or sets the learning curve showing performance vs. |
| `QueryEfficiencyRatio` | Gets or sets the query efficiency ratio. |
| `QueryStrategyName` | Gets or sets the query strategy used. |
| `SampleEfficiency` | Gets or sets the sample efficiency (accuracy gain per sample). |
| `SamplesToTargetAccuracy` | Gets or sets the number of samples needed to reach target accuracy. |
| `StoppingReason` | Gets or sets the reason why learning stopped. |
| `TargetAccuracy` | Gets or sets the target accuracy used for efficiency calculations. |
| `TotalIterations` | Gets or sets the total number of iterations completed. |
| `TotalSamplesLabeled` | Gets or sets the total number of samples labeled. |
| `TotalSelectionTime` | Gets or sets the total time spent on sample selection. |
| `TotalTime` | Gets or sets the total time for the active learning process. |
| `TotalTrainingTime` | Gets or sets the total time spent training. |

