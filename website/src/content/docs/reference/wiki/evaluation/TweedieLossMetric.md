---
title: "TweedieLossMetric<T>"
description: "Computes Tweedie Deviance Loss for regression with power parameter."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Tweedie Deviance Loss for regression with power parameter.

## For Beginners

Tweedie loss is flexible for different data types:

- Power = 0: Normal distribution (like MSE)
- Power = 1: Poisson distribution (count data)
- Power = 2: Gamma distribution (positive continuous)
- Power = 3: Inverse Gaussian distribution
- 1 < Power < 2: Compound Poisson-Gamma (insurance claims)

## How It Works

**When to use:**

- Insurance claim prediction (zeros and positives)
- Sales forecasting with many zeros
- Any data with mixed zero and positive values

