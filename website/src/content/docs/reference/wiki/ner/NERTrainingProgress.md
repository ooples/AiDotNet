---
title: "NERTrainingProgress"
description: "Reports training progress for NER models, including loss, F1 score, and epoch information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.Interfaces`

Reports training progress for NER models, including loss, F1 score, and epoch information.

## For Beginners

When training a NER model, you want to see the loss going down
and the F1 score going up over time. If the loss stops decreasing, training may have
converged. If the F1 score starts decreasing while loss keeps going down, the model
might be overfitting (memorizing training data instead of learning general patterns).

## How It Works

This class is used by `CancellationToken)` to report progress during training.
The primary metrics for NER are:

- **Loss:** How wrong the model's predictions are (lower is better). For BiLSTM-CRF models,

this is the negative log-likelihood of the correct label sequence.

- **F1 Score:** The harmonic mean of precision and recall at the entity level (higher is better).

An F1 of 0.91 on CoNLL-2003 is considered state-of-the-art for BiLSTM-CRF models.

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentBatch` | Gets or sets the current mini-batch number within the current epoch. |
| `CurrentEpoch` | Gets or sets the current epoch number (1-based). |
| `F1Score` | Gets or sets the entity-level F1 score on validation data. |
| `Loss` | Gets or sets the current training loss value. |
| `Metrics` | Gets or sets any additional metrics being tracked during training. |
| `ProgressPercentage` | Gets the overall progress as a percentage (0-100). |
| `TotalBatches` | Gets or sets the total number of mini-batches per epoch. |
| `TotalEpochs` | Gets or sets the total number of epochs planned for training. |

