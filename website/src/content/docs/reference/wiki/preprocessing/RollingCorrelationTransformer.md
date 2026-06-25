---
title: "RollingCorrelationTransformer<T>"
description: "Computes rolling correlation matrices for multivariate time series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

Computes rolling correlation matrices for multivariate time series.

## For Beginners

Correlation measures how two variables move together:

- Correlation of +1: Perfect positive relationship (when A goes up, B goes up)
- Correlation of 0: No relationship
- Correlation of -1: Perfect negative relationship (when A goes up, B goes down)

Rolling correlation shows how these relationships change over time. For example:

- Stock A and Stock B might be highly correlated during calm markets
- But become less correlated during volatile periods

This is useful for:

- Portfolio diversification (low correlation = better diversification)
- Detecting regime changes (when correlations suddenly change)
- Pair trading (betting on correlated stocks diverging temporarily)

Output for 3 features with upper triangle only:

- feature_0 vs feature_1 correlation
- feature_0 vs feature_2 correlation
- feature_1 vs feature_2 correlation

## How It Works

This transformer calculates pairwise correlations between features over rolling windows,
capturing how relationships between variables change over time.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RollingCorrelationTransformer(TimeSeriesFeatureOptions)` | Creates a new rolling correlation transformer with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsIncrementalTransform` | Gets whether this transformer supports incremental transformation. |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCorrelation(Double[],Double[])` | Computes Pearson correlation between two series. |
| `ComputeCorrelationMatrix(Tensor<>,Int32,Int32,Int32)` | Computes the correlation matrix for a rolling window. |
| `CountCorrelationPairs(Int32)` | Counts the number of correlation pairs for the given number of features. |
| `ExportParameters` | Exports transformer-specific parameters for serialization. |
| `ExtractWindow(Tensor<>,Int32,Int32,Int32)` | Extracts data for a rolling window. |
| `FitCore(Tensor<>)` |  |
| `GenerateFeatureNames` |  |
| `GetOperationNames` |  |
| `ImportParameters(Dictionary<String,Object>)` | Imports transformer-specific parameters for validation. |
| `TransformCore(Tensor<>)` |  |
| `TransformParallel(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_correlationWindowSizes` | The window sizes for correlation calculations. |
| `_fullMatrix` | Whether to output full matrix or just upper triangle. |

