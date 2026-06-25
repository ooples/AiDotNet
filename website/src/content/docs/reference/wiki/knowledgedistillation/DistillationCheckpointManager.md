---
title: "DistillationCheckpointManager<T>"
description: "Manages checkpointing during knowledge distillation training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation`

Manages checkpointing during knowledge distillation training.

## For Beginners

This class handles saving and loading model states during
distillation training. It's like the "save game" manager in a video game - it decides
when to save, what to save, and how to load progress later.

## How It Works

**Key Features:**

- Automatic checkpointing at specified intervals
- Keep only best N checkpoints based on validation metrics
- Save/restore curriculum learning progress
- Support for multi-stage distillation (student → teacher)
- Resume interrupted training

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistillationCheckpointManager(DistillationCheckpointConfig)` | Initializes a new instance of the DistillationCheckpointManager class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeleteCheckpointFiles(CheckpointMetadata)` | Deletes all files associated with a checkpoint. |
| `GetAllCheckpoints` | Gets all saved checkpoint metadata as a readonly collection. |
| `GetBestCheckpoint` | Gets the checkpoint with the best metric value. |
| `GetCheckpointByEpoch(Int32)` | Gets a checkpoint for a specific epoch. |
| `GetMostRecentCheckpoint` | Gets the most recently saved checkpoint. |
| `LoadBestCheckpoint(ICheckpointableModel,ICheckpointableModel)` | Loads the best checkpoint based on the configured metric. |
| `LoadCheckpoint(CheckpointMetadata,ICheckpointableModel,ICheckpointableModel)` | Loads a specific checkpoint. |
| `PruneOldCheckpoints` | Deletes old checkpoints, keeping only the best N. |
| `SaveCheckpoint(String,ICheckpointableModel,ICheckpointableModel,Object,CheckpointMetadata)` | Saves a checkpoint to the specified path. |
| `SaveCheckpointIfNeeded(Int32,ICheckpointableModel,ICheckpointableModel,Object,Dictionary<String,Double>,Nullable<Int32>,Boolean)` | Saves a checkpoint if conditions are met. |
| `ShouldSaveCheckpoint(Nullable<Int32>,Nullable<Int32>)` | Determines if a checkpoint should be saved based on current progress (internal logic). |

