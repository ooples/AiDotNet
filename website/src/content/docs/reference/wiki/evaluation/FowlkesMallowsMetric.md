---
title: "FowlkesMallowsMetric<T>"
description: "Computes the Fowlkes-Mallows Index: geometric mean of precision and recall."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes the Fowlkes-Mallows Index: geometric mean of precision and recall.

## For Beginners

The Fowlkes-Mallows index:

- Geometric mean of precision and recall
- Less sensitive to imbalanced data than arithmetic mean
- Range: 0 to 1, higher is better
- Often used in clustering evaluation, applicable to classification

## How It Works

FM = sqrt(PPV × TPR) = sqrt(Precision × Recall)

**Comparison to F1:**

- F1 = harmonic mean (penalizes low values more)
- FM = geometric mean (balanced between arithmetic and harmonic)

