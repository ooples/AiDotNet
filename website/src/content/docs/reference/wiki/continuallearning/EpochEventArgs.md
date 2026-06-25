---
title: "EpochEventArgs<T>"
description: "Event arguments for epoch completion events."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Interfaces`

Event arguments for epoch completion events.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EpochEventArgs(Int32,Int32,Int32,,)` | Initializes a new instance of the `EpochEventArgs` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Epoch` | Gets the epoch number (0-indexed). |
| `Loss` | Gets the training loss for this epoch. |
| `TaskId` | Gets the task identifier. |
| `TotalEpochs` | Gets the total number of epochs. |
| `ValidationLoss` | Gets the validation loss for this epoch, if available. |

