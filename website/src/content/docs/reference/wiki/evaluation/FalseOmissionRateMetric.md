---
title: "FalseOmissionRateMetric<T>"
description: "Computes False Omission Rate (FOR): FN / (FN + TN) = 1 - NPV."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes False Omission Rate (FOR): FN / (FN + TN) = 1 - NPV.

## For Beginners

FOR answers: "Of all negative predictions, how many were wrong?"

- Range: 0 to 1, lower is better
- 0 = no false negatives among negative predictions
- Important when missing positives is costly (e.g., disease screening)

## How It Works

FOR = FN / (FN + TN) = 1 - NPV

