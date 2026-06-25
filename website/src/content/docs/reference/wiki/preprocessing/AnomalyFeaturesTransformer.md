---
title: "AnomalyFeaturesTransformer<T>"
description: "Computes rolling anomaly detection features for time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

Computes rolling anomaly detection features for time series data.

## For Beginners

Anomaly detection helps find unusual or unexpected values in your data.

This is useful for:

- Fraud detection (unusual transactions)
- Equipment failure prediction (abnormal sensor readings)
- Quality control (out-of-specification products)
- Financial monitoring (unusual market movements)

The transformer creates several types of anomaly features:

- Z-score: How many standard deviations from normal
- IQR outliers: Values outside the typical range
- CUSUM: Detects gradual shifts in the data
- Isolation score: Machine learning-based anomaly scoring

## How It Works

This transformer calculates features that help identify unusual patterns,
including Z-scores, IQR-based outlier detection, CUSUM control charts, and isolation scores.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnomalyFeaturesTransformer(TimeSeriesFeatureOptions)` | Creates a new anomaly features transformer with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAnomalyFeaturesIncremental(Double[],Double,[],Int32)` | Computes anomaly features for a window of data incrementally. |
| `ComputeCusum(Double[],Double,Double)` | Computes two-sided CUSUM statistic. |
| `ComputeIncrementalFeatures(IncrementalState<>,[])` | Computes anomaly features incrementally from the circular buffer. |
| `ComputeIsolationScore(Double,Double[])` | Computes a simplified isolation score based on how easily a point can be isolated. |
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
| `_cusumH` | CUSUM decision threshold (h). |
| `_cusumK` | CUSUM sensitivity parameter (k). |
| `_enabledFeatures` | The enabled anomaly features. |
| `_iqrMultiplier` | IQR multiplier for outlier detection. |
| `_operationNames` | Cached operation names. |
| `_random` | Random number generator for isolation forest. |
| `_zScoreThreshold` | Z-score threshold for anomaly flagging. |

