---
title: "CheckpointManager<T, TInput, TOutput>"
description: "Implementation of checkpoint management for saving and restoring training state."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CheckpointManagement`

Implementation of checkpoint management for saving and restoring training state.

## How It Works

**For Beginners:** This manages checkpoints (save points) of your model training,
allowing you to save progress, resume training, and keep track of the best models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CheckpointManager(String)` | Initializes a new instance of the CheckpointManager class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CleanupKeepBest(String,Int32,MetricOptimizationDirection)` | Deletes checkpoints except the best N according to a metric. |
| `CleanupOldCheckpoints(Int32)` | Deletes old checkpoints, keeping only the most recent ones. |
| `DeleteCheckpoint(String)` | Deletes a specific checkpoint. |
| `ListCheckpoints(String,Boolean)` | Lists all available checkpoints. |
| `LoadBestCheckpoint(String,MetricOptimizationDirection)` | Loads the checkpoint with the best metric value. |
| `LoadCheckpoint(String)` | Loads a checkpoint and restores the training state. |
| `LoadLatestCheckpoint` | Loads the most recent checkpoint. |
| `SaveCheckpoint(IModel<,,>,IOptimizer<,,>,Int32,Int32,Dictionary<String,>,Dictionary<String,Object>)` | Saves a checkpoint of the current training state. |
| `TryAutoSaveCheckpoint(IModel<,,>,IOptimizer<,,>,Int32,Int32,Dictionary<String,>,Nullable<Double>,Boolean,Dictionary<String,Object>)` | Attempts to save a checkpoint automatically based on configured auto-checkpoint settings. |

