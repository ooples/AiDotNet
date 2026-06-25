---
title: "BalancedErrorRateMetric<T>"
description: "Computes Balanced Error Rate (BER): average of FNR and FPR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Balanced Error Rate (BER): average of FNR and FPR.

## For Beginners

Balanced Error Rate:

- Equal weight to errors in both classes
- Better for imbalanced datasets than standard error rate
- Range: 0 to 1, lower is better
- 0 = perfect classifier, 0.5 = random classifier

## How It Works

BER = (FNR + FPR) / 2 = 1 - Balanced Accuracy

