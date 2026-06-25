---
title: "DetectorBasedFilter<T>"
description: "Wraps any `IAnomalyDetector` for use in a preprocessing pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.OutlierHandling`

Wraps any `IAnomalyDetector` for use in a preprocessing pipeline.

## For Beginners

This class bridges the gap between anomaly detection algorithms
and data preprocessing pipelines. It lets you use any anomaly detector to clean
your data before training a model.

## How It Works

**Usage Examples:**

**Integration with PreprocessingPipeline:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DetectorBasedFilter(IAnomalyDetector<>,FilterMode,Int32[])` | Creates a new detector-based filter for preprocessing. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Detector` | Gets the underlying anomaly detector. |
| `Mode` | Gets the filter mode (how anomalies are handled). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the anomaly detector to the training data. |
| `GetAnomalyMask(Matrix<>)` | Gets a mask indicating which rows were identified as anomalies. |
| `GetAnomalyScores(Matrix<>)` | Gets the anomaly scores for each row in the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transforms flagged data by removing the flag column. |
| `TransformCore(Matrix<>)` | Transforms the data by handling anomalies according to the filter mode. |

