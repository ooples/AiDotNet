---
title: "TrainingProgress"
description: "Reports training progress for video AI models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Interfaces`

Reports training progress for video AI models.

## For Beginners

Training can take a long time. This class helps you track
progress and estimate how much longer training will take.

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentBatch` | Gets or sets the current batch number within the epoch. |
| `CurrentEpoch` | Gets or sets the current epoch number (1-based). |
| `Loss` | Gets or sets the current training loss value. |
| `Metrics` | Gets or sets any additional metrics being tracked. |
| `ProgressPercentage` | Gets the overall progress as a percentage (0-100). |
| `TotalBatches` | Gets or sets the total number of batches per epoch. |
| `TotalEpochs` | Gets or sets the total number of epochs. |

