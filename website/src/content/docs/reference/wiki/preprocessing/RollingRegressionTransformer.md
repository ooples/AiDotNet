---
title: "RollingRegressionTransformer<T>"
description: "RollingRegressionTransformer<T> — Models & Types in AiDotNet.Preprocessing.TimeSeries."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RollingRegressionTransformer(TimeSeriesFeatureOptions)` | Creates a new rolling regression transformer with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsIncrementalTransform` | Gets whether this transformer supports incremental transformation. |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCovariance(Double[],Double[],Double,Double)` | Computes sample covariance. |
| `ComputeDownsideDeviation(Double[],Double)` | Computes downside deviation (semi-deviation) for Sortino ratio. |
| `ComputeLogReturns(Double[])` | Computes log returns from prices. |
| `ComputeStdDev(Double[])` | Computes sample standard deviation. |
| `ExportParameters` | Exports transformer-specific parameters for serialization. |
| `FitCore(Tensor<>)` |  |
| `GenerateFeatureNames` |  |
| `GetOperationNames` |  |
| `GetValidPairs(Double[],Double[])` | Filters and aligns two arrays to contain only valid (non-NaN) paired values. |
| `ImportParameters(Dictionary<String,Object>)` | Imports transformer-specific parameters for validation. |
| `TransformCore(Tensor<>)` |  |
| `TransformParallel(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_annualizationFactor` | The annualization factor for ratio calculations. |
| `_benchmarkIndex` | The benchmark column index. |
| `_enabledFeatures` | The enabled regression features. |
| `_mar` | The minimum acceptable return for Sortino calculation. |
| `_operationNames` | Cached operation names. |
| `_riskFreeRate` | The period-adjusted risk-free rate. |

