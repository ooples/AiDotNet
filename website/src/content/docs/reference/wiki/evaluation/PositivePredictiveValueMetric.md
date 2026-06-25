---
title: "PositivePredictiveValueMetric<T>"
description: "Computes Positive Predictive Value (PPV): same as Precision but named differently in medical contexts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Positive Predictive Value (PPV): same as Precision but named differently in medical contexts.

## For Beginners

PPV answers: "If the test is positive, what's the probability of disease?"

- Same formula as Precision
- Common terminology in medical/diagnostic testing
- Heavily influenced by prevalence
- Even with 99% sensitivity/specificity, PPV can be low for rare diseases

## How It Works

PPV = TP / (TP + FP) = Precision

