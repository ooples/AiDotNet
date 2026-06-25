---
title: "CheckpointManagerBase<T, TInput, TOutput>"
description: "Base class for checkpoint management implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CheckpointManagement`

Base class for checkpoint management implementations.

## How It Works

**For Beginners:** This abstract base class provides common functionality for checkpoint
management systems. It handles storage path management, security validation, and JSON serialization
while leaving the specific storage implementation to derived classes.

Key features:

- Path security validation to prevent traversal attacks
- Consistent JSON serialization settings
- Thread-safe checkpoint tracking
- Auto-checkpointing configuration support

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CheckpointManagerBase(String,String)` | Initializes a new instance of the CheckpointManagerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CleanupKeepBest(String,Int32,MetricOptimizationDirection)` | Deletes checkpoints except the best N according to a metric. |
| `CleanupOldCheckpoints(Int32)` | Deletes old checkpoints, keeping only the most recent ones. |
| `ConfigureAutoCheckpointing(Int32,Int32,Boolean,String)` | Sets up automatic checkpointing during training. |
| `DeleteCheckpoint(String)` | Deletes a specific checkpoint. |
| `DeserializeFromJson(String)` | Deserializes a JSON string to an object. |
| `EnsureCheckpointDirectoryExists` | Ensures the checkpoint directory exists. |
| `GetAutoCheckpointState` | Gets the current auto-checkpointing state. |
| `GetCheckpointDirectory` | Gets the storage path for checkpoints. |
| `GetCheckpointFilePath(String)` | Generates a checkpoint file path. |
| `GetSanitizedFileName(String)` | Sanitizes a file name to prevent path traversal attacks. |
| `GetSanitizedPath(String,String)` | Gets a sanitized path, ensuring it doesn't escape the base directory. |
| `ListCheckpoints(String,Boolean)` | Lists all available checkpoints. |
| `LoadBestCheckpoint(String,MetricOptimizationDirection)` | Loads the checkpoint with the best metric value. |
| `LoadCheckpoint(String)` | Loads a checkpoint and restores the training state. |
| `LoadLatestCheckpoint` | Loads the most recent checkpoint. |
| `SaveCheckpoint(IModel<,,>,IOptimizer<,,>,Int32,Int32,Dictionary<String,>,Dictionary<String,Object>)` | Saves a checkpoint of the current training state. |
| `SerializeToJson(Object)` | Serializes an object to JSON. |
| `ShouldAutoSaveCheckpoint(Int32,Nullable<Double>,Boolean)` | Determines whether an automatic checkpoint should be saved based on current configuration. |
| `TryAutoSaveCheckpoint(IModel<,,>,IOptimizer<,,>,Int32,Int32,Dictionary<String,>,Nullable<Double>,Boolean,Dictionary<String,Object>)` | Attempts to save a checkpoint automatically based on configured auto-checkpoint settings. |
| `UpdateAutoSaveState(Int32,Nullable<Double>,Boolean)` | Updates the auto-save state after a checkpoint is saved. |
| `ValidatePathWithinDirectory(String,String)` | Validates that a path is within the specified directory. |

## Fields

| Field | Summary |
|:-----|:--------|
| `AutoConfig` | Configuration for auto-checkpointing. |
| `BestMetricValue` | The best metric value seen for improvement-based checkpointing. |
| `CheckpointDirectory` | The directory where checkpoints are stored. |
| `JsonSettings` | JSON serialization settings for consistent serialization. |
| `LastAutoSaveStep` | The last step at which an auto-checkpoint was saved. |
| `SyncLock` | Lock object for thread-safe operations. |

