---
title: "CurriculumPhaseResult<T>"
description: "Result of a single curriculum phase."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.Results`

Result of a single curriculum phase.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageNewSampleLoss` | Gets or sets the average loss for newly introduced samples. |
| `BestLoss` | Gets or sets the best loss achieved during this phase. |
| `DataFraction` | Gets or sets the data fraction used in this phase. |
| `DifficultyRange` | Gets or sets the difficulty range of samples in this phase. |
| `EndEpoch` | Gets or sets the ending epoch of this phase. |
| `EndLoss` | Gets or sets the training loss at the end of the phase. |
| `EndValidationLoss` | Gets or sets the validation loss at the end of the phase. |
| `EpochCount` | Gets the number of epochs in this phase. |
| `FinalTrainingLoss` | Gets or sets the final training loss at end of this phase. |
| `FinalValidationLoss` | Gets or sets the final validation loss at end of this phase (if available). |
| `ImprovementRate` | Gets or sets the improvement rate (improvement per epoch). |
| `LossImprovement` | Gets or sets the loss improvement in this phase. |
| `NumSamples` | Gets or sets the number of samples used in this phase. |
| `PhaseNumber` | Gets or sets the phase number (0-indexed). |
| `PhaseTimeMs` | Gets or sets the training time for this phase in milliseconds. |
| `SampleCount` | Gets or sets the number of samples available in this phase. |
| `StartEpoch` | Gets or sets the starting epoch of this phase. |
| `StartLoss` | Gets or sets the training loss at the start of the phase. |
| `StartValidationLoss` | Gets or sets the validation loss at the start of the phase. |
| `TrainingLosses` | Gets or sets the training loss history for this phase. |
| `ValidationLosses` | Gets or sets the validation loss history for this phase. |

