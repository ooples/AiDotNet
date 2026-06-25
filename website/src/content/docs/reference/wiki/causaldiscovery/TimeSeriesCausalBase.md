---
title: "TimeSeriesCausalBase<T>"
description: "Base class for time series causal discovery algorithms (Granger, PCMCI, DYNOTEARS, etc.)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery.TimeSeries`

Base class for time series causal discovery algorithms (Granger, PCMCI, DYNOTEARS, etc.).

## For Beginners

In time series, the order of events matters. These algorithms figure
out which variables help predict other variables' future values. For example, does yesterday's
stock price of company A help predict today's stock price of company B?

## How It Works

Time series causal discovery extends standard methods by considering temporal relationships.
Variable X is said to Granger-cause Y if past values of X help predict Y beyond what
Y's own past values can predict.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `MaxLag` | Maximum lag order for temporal relationships. |
| `SupportsTimeSeries` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyTimeSeriesOptions(CausalDiscoveryOptions)` | Applies time-series-specific options. |
| `ComputeRSS(Matrix<>,Vector<>,Int32,Int32)` | Computes RSS (Residual Sum of Squares) for OLS regression using generic operations. |
| `CreateLaggedData(Matrix<>,Int32,Int32)` | Creates lagged data matrix from time series: for each time step t, includes values at lags t-1, t-2, ..., t-maxLag. |

