---
title: "WeightedMAPEMetric<T>"
description: "Computes Weighted Mean Absolute Percentage Error (wMAPE): weighted by actual values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Weighted Mean Absolute Percentage Error (wMAPE): weighted by actual values.

## For Beginners

wMAPE weights errors by magnitude:

- Larger actual values contribute more to the error
- More stable than MAPE when values near zero exist
- Common in demand forecasting and retail
- Also known as WMAPE or weighted APE

## How It Works

wMAPE = 100 × Σ|y - ŷ| / Σ|y|

