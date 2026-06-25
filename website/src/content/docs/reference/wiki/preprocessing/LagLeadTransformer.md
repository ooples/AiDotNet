---
title: "LagLeadTransformer<T>"
description: "Creates lagged and leading features from time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

Creates lagged and leading features from time series data.

## For Beginners

Lagged features capture historical information:

For example, if you want to predict today's stock price:

- Lag 1: Yesterday's price
- Lag 2: Price from 2 days ago
- Lag 7: Price from a week ago

Lead features capture future values (useful for targets, not features):

- Lead 1: Tomorrow's price (what you might want to predict)

Why lag features matter:

- Many time series have autocorrelation (past values predict future)
- They help models learn temporal patterns
- They're the simplest form of "memory" for a model

Example:
Original: [100, 101, 102, 103, 104]
Lag-1: [NaN, 100, 101, 102, 103]
Lag-2: [NaN, NaN, 100, 101, 102]

## How It Works

This transformer shifts data backward (lag) or forward (lead) in time to create
features representing past or future values at each time step.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LagLeadTransformer(TimeSeriesFeatureOptions)` | Creates a new lag/lead transformer with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeIncrementalFeatures(IncrementalState<>,[])` | Computes lag/lead features incrementally from the circular buffer. |
| `ExportParameters` | Exports transformer-specific parameters for serialization. |
| `FitCore(Tensor<>)` |  |
| `GenerateFeatureNames` |  |
| `GetOperationNames` |  |
| `ImportParameters(Dictionary<String,Object>)` | Imports transformer-specific parameters for validation. |
| `TransformCore(Tensor<>)` |  |
| `TransformParallel(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lagSteps` | The lag steps to create. |
| `_leadSteps` | The lead steps to create. |

