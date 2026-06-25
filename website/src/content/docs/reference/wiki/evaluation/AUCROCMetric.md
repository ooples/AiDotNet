---
title: "AUCROCMetric<T>"
description: "Computes Area Under the ROC Curve (AUC-ROC): measures discrimination ability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Area Under the ROC Curve (AUC-ROC): measures discrimination ability.

## For Beginners

AUC-ROC answers: "If I pick a random positive and random negative,
what's the probability the model scores the positive higher?"

- AUC = 1.0: Perfect ranking
- AUC = 0.5: Random guessing (no discrimination)
- AUC < 0.5: Worse than random (model is inverted)

## How It Works

AUC-ROC measures how well the model ranks positive examples higher than negative examples.

**Advantages:** Threshold-independent, works well with imbalanced data, widely used.

