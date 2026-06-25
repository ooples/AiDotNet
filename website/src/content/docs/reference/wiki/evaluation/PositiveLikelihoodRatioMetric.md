---
title: "PositiveLikelihoodRatioMetric<T>"
description: "Computes Positive Likelihood Ratio (LR+): Sensitivity / (1 - Specificity) = TPR / FPR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Positive Likelihood Ratio (LR+): Sensitivity / (1 - Specificity) = TPR / FPR.

## For Beginners

Positive Likelihood Ratio:

- How much more likely a positive test is in disease vs healthy
- LR+ > 10: Strong evidence for disease
- LR+ 5-10: Moderate evidence
- LR+ 2-5: Weak evidence
- LR+ = 1: No diagnostic value

## How It Works

LR+ = Sensitivity / (1 - Specificity) = TPR / FPR

**Clinical use:** Multiply pre-test odds by LR+ to get post-test odds.

