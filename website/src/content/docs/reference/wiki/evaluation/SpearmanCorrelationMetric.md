---
title: "SpearmanCorrelationMetric<T>"
description: "Computes Spearman's Rank Correlation Coefficient between predictions and actuals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Spearman's Rank Correlation Coefficient between predictions and actuals.

## For Beginners

Spearman correlation measures monotonic relationships:

- Range: -1 to 1
- 1 = perfect positive monotonic relationship
- -1 = perfect negative monotonic relationship
- Non-parametric (uses ranks, not values)
- Robust to outliers unlike Pearson correlation

## How It Works

ρ = 1 - 6Σd²/(n(n²-1)) where d = rank difference

