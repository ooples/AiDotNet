---
title: "MidEpochCheckpointerOptions"
description: "Configuration options for mid-epoch checkpointing."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Pipeline`

Configuration options for mid-epoch checkpointing.

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointDirectory` | Directory to save checkpoint files. |
| `FilePrefix` | Prefix for checkpoint file names. |
| `MaxCheckpoints` | Maximum number of checkpoints to keep (oldest deleted first). |
| `SaveEveryNBatches` | Save a checkpoint every N batches. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

