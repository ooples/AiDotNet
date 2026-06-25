---
title: "ErrorRateMetric<T>"
description: "Computes Error Rate (Misclassification Rate): 1 - Accuracy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Error Rate (Misclassification Rate): 1 - Accuracy.

## For Beginners

Error rate is simply the fraction of incorrect predictions:

- Range: 0 to 1, lower is better
- 0 = perfect classifier
- Complement of accuracy

## How It Works

Error Rate = (FP + FN) / N = 1 - Accuracy

