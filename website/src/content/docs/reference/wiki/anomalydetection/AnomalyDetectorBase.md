---
title: "AnomalyDetectorBase<T>"
description: "Base class for algorithmic anomaly detectors providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.AnomalyDetection`

Base class for algorithmic anomaly detectors providing common functionality.
Extends `ModelBase` to participate in the unified
model framework (serialization, gradient computation, test generation).

## For Beginners

This base class provides shared functionality for all machine learning-based
anomaly detectors. It handles common tasks like parameter validation, threshold computation,
and distance calculations, allowing specific algorithms (like Isolation Forest or LOF) to focus
on their unique detection logic.

## How It Works

**Industry Standard Defaults:**

- Contamination: 0.1 (10%) - the expected proportion of anomalies
- Random Seed: 42 - for reproducibility

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnomalyDetectorBase(Double,Int32)` | Initializes a new instance of the `AnomalyDetectorBase` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Engine` | Provides hardware-accelerated tensor/vector operations (SIMD, GPU when available). |
| `IsFitted` |  |
| `Threshold` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `EnsureFitted` | Ensures the detector has been fitted before prediction. |
| `Fit(Matrix<>)` |  |
| `GetParameters` |  |
| `Predict(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |
| `SetParameters(Vector<>)` |  |
| `SetThresholdFromContamination(Vector<>)` | Calculates the threshold based on the contamination parameter and scores. |
| `Train(Matrix<>,Vector<>)` | Trains the anomaly detector. |
| `ValidateInput(Matrix<>,String)` | Validates that the input matrix is not null and has valid dimensions. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_contamination` | The contamination parameter representing the expected proportion of anomalies in the data. |
| `_isFitted` | Indicates whether the detector has been fitted to data. |
| `_random` | Random number generator for algorithms that require randomization. |
| `_randomSeed` | The random seed used for reproducibility. |
| `_threshold` | The threshold for classifying samples as inliers or outliers. |

