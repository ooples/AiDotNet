---
title: "MetricOptimizationDirection"
description: "Specifies the direction for metric optimization (whether lower or higher values are better)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the direction for metric optimization (whether lower or higher values are better).

## How It Works

**For Beginners:** When tracking metrics during training, you need to specify whether
you want to minimize the metric (lower is better, like loss) or maximize it (higher is better,
like accuracy). This enum lets you tell the system which direction represents improvement.

Examples:

- Loss functions: Use Minimize (you want loss to go DOWN)
- Accuracy: Use Maximize (you want accuracy to go UP)
- Error rate: Use Minimize (you want errors to go DOWN)
- F1 score: Use Maximize (you want F1 to go UP)

## Fields

| Field | Summary |
|:-----|:--------|
| `Maximize` | Higher metric values are better (e.g., accuracy, F1 score, AUC). |
| `Minimize` | Lower metric values are better (e.g., loss, error rate, MSE). |

