---
title: "DiagnosticOddsRatioMetric<T>"
description: "Computes Diagnostic Odds Ratio: (TP × TN) / (FP × FN)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Diagnostic Odds Ratio: (TP × TN) / (FP × FN).

## For Beginners

Diagnostic Odds Ratio:

- Ratio of odds of positivity in disease to odds in non-disease
- Range: 0 to infinity, higher is better
- DOR = 1 means test has no discriminative power
- DOR > 1 means positive tests are more likely in disease

## How It Works

DOR = (TP × TN) / (FP × FN) = (LR+) / (LR-)

**Medical interpretation:**

- How many times more likely a positive result is in diseased vs healthy
- Independent of disease prevalence
- Used in meta-analyses of diagnostic tests

