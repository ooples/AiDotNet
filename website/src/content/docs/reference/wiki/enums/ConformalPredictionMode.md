---
title: "ConformalPredictionMode"
description: "Defines conformal prediction calibration modes."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines conformal prediction calibration modes.

## How It Works

Conformal prediction produces prediction sets (classification) or intervals (regression) with statistical guarantees under exchangeability.
Different modes trade off compute cost, stability, and adaptivity.

## Fields

| Field | Summary |
|:-----|:--------|
| `Adaptive` | Adaptive conformal calibration that adjusts thresholds based on confidence buckets. |
| `CrossConformal` | Cross-conformal style calibration using K folds of a single calibration set. |
| `Split` | Standard split conformal calibration using a single calibration set. |

