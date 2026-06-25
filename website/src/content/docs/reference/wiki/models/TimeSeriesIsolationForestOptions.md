---
title: "TimeSeriesIsolationForestOptions<T>"
description: "Configuration options for Time Series Isolation Forest anomaly detection."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Time Series Isolation Forest anomaly detection.

## For Beginners

Isolation Forest works by randomly isolating observations.
Anomalies are easier to isolate because they are "few and different" - they end up
in shorter branches of the isolation trees.

For time series, we enhance this by considering:

- **Lag Features**: How the current value relates to recent past values
- **Rolling Statistics**: Moving averages, standard deviations
- **Seasonal Patterns**: Accounting for regular patterns like daily/weekly cycles
- **Trend**: Long-term direction of the data

This makes it effective for detecting:

- Sudden spikes or drops (point anomalies)
- Values that are unusual given the context (contextual anomalies)
- Unusual patterns over time (collective anomalies)

## How It Works

Time Series Isolation Forest extends the classic Isolation Forest algorithm to handle
temporal data by incorporating lag features, rolling statistics, and seasonal decomposition.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesIsolationForestOptions` | Creates a new instance with default values. |
| `TimeSeriesIsolationForestOptions(TimeSeriesIsolationForestOptions<>)` | Creates a copy of the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContaminationRate` | Gets or sets the expected proportion of anomalies in the dataset. |
| `LagFeatures` | Gets or sets the number of lag features to include. |
| `MaxDepth` | Gets or sets the maximum depth of each isolation tree. |
| `NumTrees` | Gets or sets the number of isolation trees in the forest. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `RollingWindowSize` | Gets or sets the window size for rolling statistics (mean, std, min, max). |
| `SampleSize` | Gets or sets the number of samples to use when building each tree. |
| `UseSeasonalDecomposition` | Gets or sets whether to decompose the series into trend and seasonal components. |
| `UseTrendFeatures` | Gets or sets whether to include trend-based features. |

