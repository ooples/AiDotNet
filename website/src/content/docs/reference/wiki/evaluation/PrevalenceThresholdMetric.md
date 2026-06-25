---
title: "PrevalenceThresholdMetric<T>"
description: "Computes Prevalence Threshold: the prevalence at which the test would have 50% PPV."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Prevalence Threshold: the prevalence at which the test would have 50% PPV.

## For Beginners

Prevalence Threshold tells you the minimum disease prevalence needed:

- Below this prevalence, PPV drops below 50% (more false positives than true positives)
- Lower is better (test is useful even for rare conditions)
- Useful for evaluating screening tests for rare diseases

## How It Works

PT = sqrt(FPR) / (sqrt(TPR) + sqrt(FPR))

