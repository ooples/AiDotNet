---
title: "NegativeLikelihoodRatioMetric<T>"
description: "Computes Negative Likelihood Ratio (LR-): (1 - Sensitivity) / Specificity = FNR / TNR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Negative Likelihood Ratio (LR-): (1 - Sensitivity) / Specificity = FNR / TNR.

## For Beginners

Negative Likelihood Ratio:

- How much less likely a negative test is in disease vs healthy
- LR- < 0.1: Strong evidence against disease
- LR- 0.1-0.2: Moderate evidence
- LR- 0.2-0.5: Weak evidence
- LR- = 1: No diagnostic value

## How It Works

LR- = (1 - Sensitivity) / Specificity = FNR / TNR

**Clinical use:** Multiply pre-test odds by LR- to get post-test odds after negative result.

