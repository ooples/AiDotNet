---
title: "ICheckpointManager<T, TInput, TOutput>"
description: "Defines the contract for checkpoint management systems that save and restore training state."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for checkpoint management systems that save and restore training state.

## How It Works

A checkpoint manager handles saving and restoring the complete state of model training,
allowing you to pause and resume training, recover from failures, and track model evolution.

**For Beginners:** Think of checkpoints like save points in a video game. They let you:

- Save your progress so you can come back later
- Go back to an earlier point if something goes wrong
- Keep the best version you've found so far

Checkpoints typically save:

- Model parameters (weights and biases)
- Optimizer state (momentum, learning rate schedule, etc.)
- Training metadata (epoch number, step count)
- Performance metrics

Why checkpoint management matters:

- Training can be interrupted (crashes, time limits)
- You want to keep the best model even if later training makes it worse
- Long training runs need progress saved periodically
- Enables experimentation with different training strategies from same point

## Methods

| Method | Summary |
|:-----|:--------|
| `CleanupKeepBest(String,Int32,MetricOptimizationDirection)` | Deletes checkpoints except the best N according to a metric. |
| `CleanupOldCheckpoints(Int32)` | Deletes old checkpoints, keeping only a specified number of the most recent ones. |
| `ConfigureAutoCheckpointing(Int32,Int32,Boolean,String)` | Sets up automatic checkpointing during training. |
| `DeleteCheckpoint(String)` | Deletes a specific checkpoint. |
| `GetCheckpointDirectory` | Gets the storage path for checkpoints. |
| `ListCheckpoints(String,Boolean)` | Lists all available checkpoints. |
| `LoadBestCheckpoint(String,MetricOptimizationDirection)` | Loads the checkpoint with the best metric value. |
| `LoadCheckpoint(String)` | Loads a checkpoint and restores the training state. |
| `LoadLatestCheckpoint` | Loads the most recent checkpoint. |
| `SaveCheckpoint(IModel<,,>,IOptimizer<,,>,Int32,Int32,Dictionary<String,>,Dictionary<String,Object>)` | Saves a checkpoint of the current training state. |
| `ShouldAutoSaveCheckpoint(Int32,Nullable<Double>,Boolean)` | Determines whether an automatic checkpoint should be saved based on current configuration. |
| `TryAutoSaveCheckpoint(IModel<,,>,IOptimizer<,,>,Int32,Int32,Dictionary<String,>,Nullable<Double>,Boolean,Dictionary<String,Object>)` | Attempts to save a checkpoint automatically based on configured auto-checkpoint settings. |
| `UpdateAutoSaveState(Int32,Nullable<Double>,Boolean)` | Updates the auto-save state after a checkpoint is saved. |

