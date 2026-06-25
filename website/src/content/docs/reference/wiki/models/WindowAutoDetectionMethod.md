---
title: "WindowAutoDetectionMethod"
description: "Methods for auto-detecting optimal window sizes."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Methods for auto-detecting optimal window sizes.

## Fields

| Field | Summary |
|:-----|:--------|
| `Autocorrelation` | Use autocorrelation function (ACF) to detect seasonality. |
| `GridSearch` | Use grid search with cross-validation to find best windows. |
| `Heuristic` | Use simple heuristic rules based on data characteristics. |
| `SpectralAnalysis` | Use spectral analysis (FFT) to detect dominant frequencies. |

