---
title: "AveragingMethod"
description: "Specifies the averaging method for multi-class/multi-label classification metrics."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Enums`

Specifies the averaging method for multi-class/multi-label classification metrics.

## For Beginners

Imagine you have a model that classifies images into 3 categories:
cats, dogs, and birds. Each category has its own precision/recall. The averaging method
determines how to combine these into a single score:

- **Micro:** Treat all samples equally (good when classes are balanced)
- **Macro:** Treat all classes equally (good for imbalanced data)
- **Weighted:** Weight by class frequency (compromise between micro/macro)

## How It Works

When computing metrics like precision, recall, or F1-score for multi-class problems,
there are different ways to aggregate per-class values into a single number.

## Fields

| Field | Summary |
|:-----|:--------|
| `Binary` | Binary: Only report results for the positive class (class 1). |
| `Macro` | Macro-averaging: Compute metric for each class, then take unweighted mean. |
| `Micro` | Micro-averaging: Aggregate contributions of all classes to compute the metric. |
| `None` | No averaging: Return a score for each class individually. |
| `Samples` | Samples-averaging: For multi-label classification, compute metric for each sample and average across samples. |
| `Weighted` | Weighted-averaging: Compute metric for each class, then take weighted mean by support. |

