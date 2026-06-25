---
title: "CurriculumLearningResult<T>"
description: "Result of curriculum learning training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.Results`

Result of curriculum learning training.

## For Beginners

This class contains all the information about how
curriculum learning training went - the final model performance, how long it took,
and detailed information about each curriculum phase.

## How It Works

**Key Metrics:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AveragePhaseImprovement` | Gets or sets the average improvement per phase. |
| `BestEpoch` | Gets or sets the epoch at which best validation loss was achieved. |
| `BestTrainingLoss` | Gets or sets the best training loss achieved during training. |
| `BestValidationLoss` | Gets or sets the best validation loss achieved during training. |
| `CurriculumEfficiency` | Gets or sets the curriculum efficiency metric. |
| `CurriculumProgression` | Gets or sets the curriculum progression history. |
| `EarlyStopEpoch` | Gets or sets the epoch at which early stopping was triggered. |
| `EarlyStopTriggered` | Gets or sets whether early stopping was triggered. |
| `ErrorMessage` | Gets or sets the error message if training failed. |
| `FinalDifficulties` | Gets or sets the final difficulty scores for all samples. |
| `FinalTrainingLoss` | Gets or sets the final training loss. |
| `FinalValidationLoss` | Gets or sets the final validation loss (if validation data provided). |
| `PhaseResults` | Gets or sets the results for each curriculum phase. |
| `PhasesCompleted` | Gets or sets the number of curriculum phases completed. |
| `SchedulerStatistics` | Gets or sets the scheduler statistics at the end of training. |
| `Success` | Gets or sets whether training was successful. |
| `TotalEpochs` | Gets or sets the total number of epochs trained. |
| `TotalSamplesUsed` | Gets or sets the total number of samples used during training. |
| `TrainingLossHistory` | Gets or sets the training history (loss per epoch). |
| `TrainingSamples` | Gets or sets the total number of training samples. |
| `TrainingTimeFormatted` | Gets the training time in human-readable format. |
| `TrainingTimeMs` | Gets or sets the total training time in milliseconds. |
| `ValidationLossHistory` | Gets or sets the validation loss history. |
| `ValidationSamples` | Gets or sets the total number of validation samples (if validation data provided). |

