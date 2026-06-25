---
title: "CheckpointMetricType"
description: "Standard metrics for checkpoint selection and early stopping."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Standard metrics for checkpoint selection and early stopping.

## For Beginners

Use Loss for most cases, or Accuracy for classification.
The metric determines when to save checkpoints and when to stop early.

## How It Works

These metrics determine which checkpoint is considered "best" during training.

## Fields

| Field | Summary |
|:-----|:--------|
| `Accuracy` | Accuracy metric (higher is better). |
| `BLEU` | BLEU score (higher is better). |
| `Custom` | Custom metric defined by user. |
| `F1Score` | F1 score (higher is better). |
| `Loss` | Training or validation loss (lower is better). |
| `Perplexity` | Perplexity (lower is better). |
| `ROUGE` | ROUGE score (higher is better). |
| `RewardScore` | Reward model score (higher is better). |
| `WinRate` | Win rate against reference (higher is better). |

