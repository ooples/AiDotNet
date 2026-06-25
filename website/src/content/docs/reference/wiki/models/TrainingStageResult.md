---
title: "TrainingStageResult<T, TInput, TOutput>"
description: "Result of executing a training stage."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Options`

Result of executing a training stage.

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointPath` | Gets or sets the checkpoint path if one was saved. |
| `Duration` | Gets or sets the duration of this stage. |
| `ErrorMessage` | Gets or sets any error message if the stage failed. |
| `EvaluationMetrics` | Gets or sets evaluation metrics if evaluation was performed. |
| `Metrics` | Gets or sets the training metrics from this stage. |
| `Model` | Gets or sets the model after this stage. |
| `StageIndex` | Gets or sets the stage index in the pipeline. |
| `StageName` | Gets or sets the name of the completed stage. |
| `Success` | Gets or sets whether the stage completed successfully. |

