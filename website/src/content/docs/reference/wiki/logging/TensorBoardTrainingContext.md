---
title: "TensorBoardTrainingContext"
description: "Context manager for training runs with automatic TensorBoard logging."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Logging`

Context manager for training runs with automatic TensorBoard logging.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TensorBoardTrainingContext(String,String,Dictionary<String,Object>)` | Creates a new training context. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Elapsed` | Gets elapsed time since context creation. |
| `GlobalStep` | Gets or sets the current global step. |
| `Writer` | Gets the underlying SummaryWriter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Releases resources. |
| `LogElapsedTime` | Logs elapsed time. |
| `LogModelWeights(Dictionary<String,Single[]>,Dictionary<String,Single[]>)` | Logs model weights at current step. |
| `LogTrainStep(Single,Nullable<Single>,Nullable<Single>)` | Logs a training step with automatic step incrementing. |
| `LogValStep(Single,Nullable<Single>)` | Logs a validation step (does not increment global step). |

