---
title: "FalsePositiveRateMetric<T>"
description: "Computes False Positive Rate (FPR): proportion of actual negatives incorrectly predicted as positive."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes False Positive Rate (FPR): proportion of actual negatives incorrectly predicted as positive.

## For Beginners

FPR answers: "Of all the actual negatives, how many did I incorrectly flag as positive?"

- FPR = 0: Perfect - no false alarms
- FPR = 0.1: 10% of actual negatives are incorrectly flagged as positive
- FPR = 1: All negatives are incorrectly classified as positive

Critical in spam detection (low FPR = fewer legitimate emails marked as spam)
and medical screening (low FPR = fewer healthy people incorrectly told they're sick).

## How It Works

FPR = FP / (FP + TN) = 1 - Specificity

