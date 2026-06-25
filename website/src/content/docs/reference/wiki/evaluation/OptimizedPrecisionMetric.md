---
title: "OptimizedPrecisionMetric<T>"
description: "Computes Optimized Precision: Accuracy - |Sensitivity - Specificity| / (Sensitivity + Specificity)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Optimized Precision: Accuracy - |Sensitivity - Specificity| / (Sensitivity + Specificity).

## For Beginners

Optimized Precision:

- Penalizes classifiers that favor one class over another
- Encourages balance between sensitivity and specificity
- Range: typically 0 to 1
- Higher values indicate better overall performance

## How It Works

OP = Accuracy - |Sensitivity - Specificity| / (Sensitivity + Specificity)

