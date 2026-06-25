---
title: "CheckpointMetadata"
description: "Metadata about a saved checkpoint."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation`

Metadata about a saved checkpoint.

## Properties

| Property | Summary |
|:-----|:--------|
| `Batch` | Batch number when checkpoint was saved (if applicable). |
| `BatchCounter` | Batch counter value when checkpoint was saved (for resume continuity). |
| `Epoch` | Epoch number when checkpoint was saved. |
| `FilePath` | Base file path for this checkpoint. |
| `Metrics` | Training/validation metrics at checkpoint time. |
| `StrategyCheckpointPath` | Path to strategy state file. |
| `StudentCheckpointPath` | Path to student model checkpoint file. |
| `TeacherCheckpointPath` | Path to teacher model checkpoint file. |
| `Timestamp` | Timestamp when checkpoint was created. |

