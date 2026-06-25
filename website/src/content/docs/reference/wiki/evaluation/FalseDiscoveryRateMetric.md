---
title: "FalseDiscoveryRateMetric<T>"
description: "Computes False Discovery Rate (FDR): FP / (FP + TP) = 1 - Precision."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes False Discovery Rate (FDR): FP / (FP + TP) = 1 - Precision.

## For Beginners

FDR answers: "Of all positive predictions, how many were wrong?"

- Range: 0 to 1, lower is better
- 0 = no false positives among positive predictions
- Important in scientific testing (controlling false discoveries)

## How It Works

FDR = FP / (FP + TP) = 1 - Precision = 1 - PPV

