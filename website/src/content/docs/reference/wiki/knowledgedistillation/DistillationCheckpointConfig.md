---
title: "DistillationCheckpointConfig"
description: "Configuration for distillation checkpoint management."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.KnowledgeDistillation`

Configuration for distillation checkpoint management.

## For Beginners

This class controls when and how models are saved during
knowledge distillation training. Think of it like the "auto-save" settings in a video game.

## Properties

| Property | Summary |
|:-----|:--------|
| `BestMetric` | Metric to use for determining "best" checkpoint (e.g., "validation_loss", "accuracy"). |
| `CheckpointDirectory` | Directory where checkpoints will be saved. |
| `CheckpointPrefix` | Prefix for checkpoint filenames. |
| `KeepBestN` | Keep only the best N checkpoints based on validation metric. |
| `LowerIsBetter` | Whether lower metric values are better (true for loss, false for accuracy). |
| `SaveCurriculumState` | Save curriculum strategy state with checkpoint. |
| `SaveEveryBatches` | Save checkpoint every N batches (0 = disabled). |
| `SaveEveryEpochs` | Save checkpoint every N epochs (0 = disabled). |
| `SaveMetadata` | Save training metadata (epoch number, loss history, etc.). |
| `SaveStudent` | Save the student model checkpoint. |
| `SaveTeacher` | Save the teacher model checkpoint. |

