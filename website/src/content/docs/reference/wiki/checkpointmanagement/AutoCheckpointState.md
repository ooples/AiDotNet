---
title: "AutoCheckpointState"
description: "Represents the current state of auto-checkpointing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CheckpointManagement`

Represents the current state of auto-checkpointing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoCheckpointState(Boolean,Int32,Boolean,Int32,String,Int32,Nullable<Double>)` | Initializes a new AutoCheckpointState. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestMetricValue` | Best metric value seen so far. |
| `IsEnabled` | Whether auto-checkpointing is enabled. |
| `KeepLast` | Number of recent checkpoints to keep. |
| `LastSaveStep` | Last step at which a checkpoint was saved. |
| `MetricName` | Metric name for improvement tracking. |
| `SaveFrequency` | Save frequency in steps. |
| `SaveOnImprovement` | Whether saving on improvement is enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` |  |

