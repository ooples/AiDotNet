---
title: "HyperparameterApplicationResult"
description: "Contains the results of applying agent-recommended hyperparameters to a model's options."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Contains the results of applying agent-recommended hyperparameters to a model's options.

## For Beginners

When the AI agent recommends hyperparameters, this class tells you
exactly what happened when those recommendations were applied to your model:

- **Applied**: Parameters that were successfully set on the model
- **Skipped**: Parameters the agent recommended but the model doesn't support
- **Failed**: Parameters that couldn't be set due to errors (e.g., wrong data type)
- **Warnings**: Issues that didn't prevent application but may need attention (e.g., values outside typical ranges)

## How It Works

This class tracks which hyperparameters were successfully applied, which were skipped
(no matching property found), which failed (type conversion or other errors), and any
warnings generated during the process (e.g., out-of-range values).

## Properties

| Property | Summary |
|:-----|:--------|
| `Applied` | Gets the dictionary of successfully applied hyperparameters (parameter name -> value set). |
| `Failed` | Gets the dictionary of failed hyperparameters (parameter name -> error message). |
| `HasAppliedParameters` | Gets a value indicating whether any parameters were successfully applied. |
| `Skipped` | Gets the dictionary of skipped hyperparameters (parameter name -> value, no matching property found). |
| `Warnings` | Gets the list of warning messages generated during hyperparameter application. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Gets a human-readable summary of the hyperparameter application results. |

