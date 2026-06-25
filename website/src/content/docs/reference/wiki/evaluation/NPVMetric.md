---
title: "NPVMetric<T>"
description: "Computes Negative Predictive Value (NPV): proportion of negative predictions that are correct."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Negative Predictive Value (NPV): proportion of negative predictions that are correct.

## For Beginners

NPV answers: "When my model predicts negative, how often is it right?"

- NPV = 1: Every negative prediction is correct (no false negatives among negative predictions)
- NPV = 0.8: 80% of negative predictions are correct

Important in medical diagnostics: high NPV means you can trust a negative test result.
Related to but different from Specificity (TN / (TN + FP)).

## How It Works

NPV = TN / (TN + FN)

