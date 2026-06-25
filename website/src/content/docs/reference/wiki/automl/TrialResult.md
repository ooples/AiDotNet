---
title: "TrialResult"
description: "Represents the result of a single trial during AutoML search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

Represents the result of a single trial during AutoML search.

## For Beginners

AutoML runs many small experiments called "trials". Each trial:

- Tries one model family with one set of settings.
- Trains the model and scores it.
- Records whether it succeeded and how well it did.

This class stores the outcome of one of those experiments.

## How It Works

This type is used by internal AutoML implementations to track trial execution outcomes. When accessed via public
APIs (for example, through `GetTrialHistory`),
sensitive fields like raw hyperparameter values must be redacted to align with the AiDotNet facade/IP goals.

## Properties

| Property | Summary |
|:-----|:--------|
| `CandidateModelType` | Gets or sets the candidate model family used for the trial, when known. |
| `Duration` | Gets or sets the duration of the trial. |
| `ErrorMessage` | Gets or sets an error message if the trial failed. |
| `Metadata` | Gets or sets additional metadata about the trial. |
| `Parameters` | Gets or sets the hyperparameters used in this trial. |
| `Score` | Gets or sets the score achieved by this trial. |
| `Success` | Gets or sets a value indicating whether the trial completed successfully. |
| `Timestamp` | Gets or sets the UTC timestamp when the trial was completed. |
| `TrialId` | Gets or sets the unique identifier for the trial. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of this trial result including sensitive parameters. |
| `CloneRedacted` | Creates a deep copy of this trial result with sensitive fields redacted. |

