---
title: "AutoMLTrialSummary"
description: "Represents a redacted (safe-to-share) summary of a single AutoML trial."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents a redacted (safe-to-share) summary of a single AutoML trial.

## For Beginners

AutoML tries many different "trials" (model attempts).
This class records what happened for one trial:

- Did it succeed?
- How long did it take?
- What score did it achieve?

## How It Works

This summary intentionally excludes hyperparameter values, model weights, and other implementation details.
It is designed to support the AiDotNet facade pattern, where users can review outcomes without gaining access
to proprietary configuration details.

## Properties

| Property | Summary |
|:-----|:--------|
| `CompletedUtc` | Gets or sets the timestamp when the trial completed (UTC). |
| `Duration` | Gets or sets the duration of the trial. |
| `ErrorMessage` | Gets or sets an error message when the trial fails. |
| `Score` | Gets or sets the score achieved by this trial. |
| `Success` | Gets or sets a value indicating whether the trial completed successfully. |
| `TrialId` | Gets or sets the unique identifier for the trial. |

