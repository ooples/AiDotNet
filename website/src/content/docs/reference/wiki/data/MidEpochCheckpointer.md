---
title: "MidEpochCheckpointer"
description: "Saves and restores training state mid-epoch for fault tolerance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Pipeline`

Saves and restores training state mid-epoch for fault tolerance.

## How It Works

Enables resuming training from the exact batch within an epoch after a failure.
Saves the current batch index, epoch number, and any custom state as a binary checkpoint.
Automatically rotates old checkpoints to limit disk usage.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MidEpochCheckpointer(MidEpochCheckpointerOptions)` | Creates a new mid-epoch checkpointer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadCheckpoint(String)` | Loads a specific checkpoint file. |
| `LoadLatestCheckpoint` | Loads the latest checkpoint from the checkpoint directory. |
| `OnBatchComplete(Int32,Int32,Byte[])` | Called after each batch to potentially save a checkpoint. |
| `SaveCheckpoint(Int32,Int32,Byte[])` | Saves a checkpoint with the current training state. |

