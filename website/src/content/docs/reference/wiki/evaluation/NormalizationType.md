---
title: "NormalizationType"
description: "Specifies the normalization method for confusion matrices and metrics."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Enums`

Specifies the normalization method for confusion matrices and metrics.

## For Beginners

A raw confusion matrix shows counts (e.g., "50 true positives").
Normalized versions show proportions, which are easier to interpret:

- **None:** Raw counts
- **ByTrue:** "What percentage of actual positives were correctly identified?"
- **ByPredicted:** "What percentage of predicted positives were correct?"
- **All:** "What percentage of all samples fall into each cell?"

## How It Works

Confusion matrix normalization helps interpret results, especially with imbalanced classes.
Different normalizations answer different questions.

## Fields

| Field | Summary |
|:-----|:--------|
| `All` | Normalize by total: All cells sum to 1. |
| `ByPredicted` | Normalize by predicted labels (columns): Each column sums to 1. |
| `ByTrue` | Normalize by true labels (rows): Each row sums to 1. |
| `Log` | Log normalization: Apply log transformation. |
| `MinMax` | Min-max normalization: Scale values to [0, 1] range. |
| `None` | No normalization: Raw counts in confusion matrix. |
| `Robust` | Robust normalization: Use median and IQR instead of mean and std. |
| `ZScore` | Z-score (standard) normalization: Transform to mean=0, std=1. |

