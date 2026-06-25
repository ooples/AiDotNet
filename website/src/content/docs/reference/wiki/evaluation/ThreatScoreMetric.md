---
title: "ThreatScoreMetric<T>"
description: "Computes Threat Score (Critical Success Index): TP / (TP + FN + FP)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Threat Score (Critical Success Index): TP / (TP + FN + FP).

## For Beginners

Threat Score (also called Critical Success Index):

- Measures overlap between predictions and actuals
- Ignores True Negatives (TN)
- Range: 0 to 1, higher is better
- Used heavily in meteorology (weather forecasting)

## How It Works

TS = TP / (TP + FN + FP)

**When to use:**

- When True Negatives are common and not interesting
- When you care about detecting rare events
- Weather prediction, rare disease detection

