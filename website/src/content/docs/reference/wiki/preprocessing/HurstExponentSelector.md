---
title: "HurstExponentSelector<T>"
description: "Hurst Exponent based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Complexity`

Hurst Exponent based Feature Selection.

## For Beginners

The Hurst exponent H indicates trend behavior:
H > 0.5 means trending (persistent) - ups followed by ups, downs by downs;
H < 0.5 means mean-reverting (anti-persistent) - ups followed by downs;
H = 0.5 means random walk (no memory). Financial time series with H ≠ 0.5
may contain exploitable patterns.

## How It Works

Selects features based on their Hurst exponent, which measures
long-range dependence and self-similarity in time series.

