---
title: "HammingLossMetric<T>"
description: "Computes Hamming Loss: fraction of labels that are incorrectly predicted."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Hamming Loss: fraction of labels that are incorrectly predicted.

## For Beginners

Hamming loss is the complement of accuracy (1 - accuracy).
It tells you what fraction of your predictions are wrong.

- Hamming Loss = 0: Perfect predictions (all correct)
- Hamming Loss = 0.3: 30% of predictions are wrong
- Hamming Loss = 1: All predictions are wrong

Particularly useful for multi-label classification.

## How It Works

Hamming Loss = (1/N) * Σ(prediction ≠ actual)

