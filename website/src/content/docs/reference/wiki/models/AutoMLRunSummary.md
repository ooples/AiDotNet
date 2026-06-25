---
title: "AutoMLRunSummary"
description: "Represents a redacted (safe-to-share) summary of an AutoML run."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents a redacted (safe-to-share) summary of an AutoML run.

## For Beginners

AutoML is an automatic process that tries multiple model attempts ("trials")
and keeps the best one. This class tells you how that search went, including:

- How many trials ran
- The best score found
- A per-trial outcome history (without exposing secret settings)

## How It Works

This summary is intended for facade outputs (for example, `AiModelResult`) and avoids exposing
hyperparameters or sensitive model details. It provides transparency into the AutoML process while protecting
proprietary implementation choices.

## Properties

| Property | Summary |
|:-----|:--------|
| `BestScore` | Gets or sets the best score achieved during the AutoML search. |
| `EnsembleSize` | Gets or sets the number of models in the selected ensemble, when applicable. |
| `MaximizeMetric` | Gets or sets a value indicating whether higher metric values are better. |
| `NASResult` | Gets or sets NAS-specific result information when a NAS strategy was used. |
| `OptimizationMetric` | Gets or sets the optimization metric used to rank trials. |
| `SearchEndedUtc` | Gets or sets the UTC timestamp when the AutoML search ended. |
| `SearchStartedUtc` | Gets or sets the UTC timestamp when the AutoML search started. |
| `SearchStrategy` | Gets or sets the search strategy used for the AutoML run, if known. |
| `TimeLimit` | Gets or sets the time limit used for the AutoML search. |
| `TrialLimit` | Gets or sets the maximum number of trials allowed for the AutoML search. |
| `Trials` | Gets or sets a redacted list of trial summaries. |
| `UsedEnsemble` | Gets or sets a value indicating whether AutoML selected an ensemble as the final model. |

