---
title: "RLStepMetrics<T>"
description: "Metrics for a single RL training step."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Configuration`

Metrics for a single RL training step.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RLStepMetrics` | Creates a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DidTrain` | Whether training occurred this step. |
| `Episode` | Current episode number. |
| `Loss` | Training loss (if training occurred this step). |
| `Reward` | Reward received for this step. |
| `Step` | Step number within the current episode. |
| `TotalSteps` | Total steps across all episodes. |

