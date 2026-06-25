---
title: "AUCPRMetric<T>"
description: "Computes Area Under the Precision-Recall Curve (AUC-PR)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Area Under the Precision-Recall Curve (AUC-PR).

## For Beginners

AUC-PR measures the trade-off between precision and recall:

- Better than AUC-ROC for imbalanced datasets
- Focuses on the positive class performance
- Range: 0 to 1, higher is better
- Baseline depends on class imbalance (unlike ROC's 0.5)

## How It Works

**When to use AUC-PR vs AUC-ROC:**

- AUC-PR: When positive class is rare and important (fraud, disease)
- AUC-ROC: When both classes are equally important

