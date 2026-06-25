---
title: "TrueNegativeRateMetric<T>"
description: "Computes True Negative Rate (TNR): same as Specificity, TN / (TN + FP)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes True Negative Rate (TNR): same as Specificity, TN / (TN + FP).

## For Beginners

TNR answers: "Of all actual negatives, what proportion did we correctly identify?"
This is the same as Specificity but named from a different perspective.

- TNR = 1: All negatives correctly identified
- TNR = 0.9: 90% of negatives correctly identified, 10% false alarms

## How It Works

TNR = TN / (TN + FP) = Specificity = 1 - FPR

