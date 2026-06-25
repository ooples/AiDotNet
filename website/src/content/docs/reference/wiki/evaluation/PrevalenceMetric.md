---
title: "PrevalenceMetric<T>"
description: "Computes Prevalence: fraction of actual positives in the dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Prevalence: fraction of actual positives in the dataset.

## For Beginners

Prevalence is the base rate of the positive class:

- Range: 0 to 1
- Not a performance metric but important context
- Affects interpretation of other metrics like PPV
- Low prevalence makes PPV unreliable even with high sensitivity

## How It Works

Prevalence = (TP + FN) / N = Positives / Total

