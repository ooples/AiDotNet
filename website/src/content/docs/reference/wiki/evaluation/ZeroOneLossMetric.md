---
title: "ZeroOneLossMetric<T>"
description: "Computes Zero-One Loss: fraction of misclassifications (complement of accuracy)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Zero-One Loss: fraction of misclassifications (complement of accuracy).

## For Beginners

Zero-one loss simply counts the proportion of wrong predictions.
It's exactly 1 - Accuracy. Unlike other loss functions, it treats all misclassifications equally
regardless of how confident the wrong prediction was.

## How It Works

Zero-One Loss = (1/N) * Σ(prediction ≠ actual)

