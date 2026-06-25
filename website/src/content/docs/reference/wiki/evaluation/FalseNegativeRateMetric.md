---
title: "FalseNegativeRateMetric<T>"
description: "Computes False Negative Rate (FNR): proportion of actual positives incorrectly predicted as negative."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes False Negative Rate (FNR): proportion of actual positives incorrectly predicted as negative.

## For Beginners

FNR answers: "Of all actual positives, how many did we miss?"

- FNR = 0: Perfect - caught all positives
- FNR = 0.2: Missed 20% of actual positives
- FNR = 1: Missed all positives

Critical in medical diagnostics (low FNR = fewer missed diseases) and fraud detection
(low FNR = fewer missed fraudulent transactions).

## How It Works

FNR = FN / (FN + TP) = 1 - Recall = 1 - TPR

