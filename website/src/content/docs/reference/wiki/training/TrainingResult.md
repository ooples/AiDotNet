---
title: "TrainingResult<T>"
description: "Contains the results of a training run, including the trained model, loss history, and metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Training`

Contains the results of a training run, including the trained model, loss history, and metadata.

## For Beginners

After training completes, this object tells you everything about how it went:
the trained model ready for predictions, the loss values at each epoch (so you can see if training
improved over time), how long it took, and whether it completed successfully.

## Properties

| Property | Summary |
|:-----|:--------|
| `Completed` | Gets or sets whether the training run completed without error. |
| `EpochLosses` | Gets or sets the loss value recorded at the end of each epoch. |
| `TotalEpochs` | Gets or sets the total number of epochs that were executed. |
| `TrainedModel` | Gets or sets the trained model ready for making predictions. |
| `TrainingDuration` | Gets or sets the total wall-clock time spent training. |

