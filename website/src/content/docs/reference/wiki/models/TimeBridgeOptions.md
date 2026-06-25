---
title: "TimeBridgeOptions<T>"
description: "Configuration options for TimeBridge (Non-Stationarity Matters for Time Series Foundation Models)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TimeBridge (Non-Stationarity Matters for Time Series Foundation Models).

## How It Works

TimeBridge addresses the critical non-stationarity gap in time series foundation models.
It introduces a bridge mechanism that preserves and restores non-stationary information
(trends, level shifts) that is typically lost during standard normalization.

**Reference:** "TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `BridgeDimension` | Gets or sets the dimension of the non-stationarity bridge module. |
| `UseStationarityGating` | Gets or sets whether to use stationarity gating for adaptive restoration. |

