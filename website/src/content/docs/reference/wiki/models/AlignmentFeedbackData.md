---
title: "AlignmentFeedbackData<T>"
description: "Contains human feedback data for AI alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Contains human feedback data for AI alignment.

## Properties

| Property | Summary |
|:-----|:--------|
| `Inputs` | Gets or sets the input prompts or examples. |
| `Outputs` | Gets or sets the model outputs for each input. |
| `Preferences` | Gets or sets human preference comparisons. |
| `Ratings` | Gets or sets numerical ratings for each output (optional). |
| `Rewards` | Gets or sets reward labels for reinforcement learning. |
| `TextualFeedback` | Gets or sets textual feedback for outputs (optional). |

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsurePreferencesValid` | Validates preference indices and throws if invalid. |
| `ValidatePreferences` | Validates that preference indices are within valid bounds. |

