---
title: "HyperparameterTrial<T>"
description: "Represents a single trial in hyperparameter optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents a single trial in hyperparameter optimization.

## How It Works

**For Beginners:** A trial is one attempt at training with a specific set of hyperparameters.
The optimizer tries many trials to find the best hyperparameter values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperparameterTrial(Int32)` | Initializes a new instance of the HyperparameterTrial class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EndTime` | Gets or sets the end time of the trial. |
| `IntermediateValues` | Gets or sets intermediate values logged during training. |
| `ObjectiveValue` | Gets or sets the objective value achieved (e.g., validation accuracy). |
| `Parameters` | Gets or sets the hyperparameter values for this trial. |
| `StartTime` | Gets or sets the start time of the trial. |
| `Status` | Gets or sets the status of the trial. |
| `SystemAttributes` | Gets or sets system attributes for the trial. |
| `TrialId` | Gets the unique identifier for this trial. |
| `TrialNumber` | Gets the trial number (sequential). |
| `UserAttributes` | Gets or sets user attributes for the trial. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Complete()` | Marks the trial as complete with the final objective value. |
| `Fail` | Marks the trial as failed. |
| `GetDuration` | Gets the duration of the trial. |
| `Prune` | Marks the trial as pruned (stopped early). |
| `ReportIntermediateValue(Int32,)` | Reports an intermediate value for a training step. |

