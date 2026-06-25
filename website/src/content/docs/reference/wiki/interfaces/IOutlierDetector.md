---
title: "IOutlierDetector<T>"
description: "Defines methods for algorithmic outlier/anomaly detection using a fit-predict pattern."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines methods for algorithmic outlier/anomaly detection using a fit-predict pattern.

## For Beginners

This interface provides a machine learning-style approach to outlier detection.
Unlike simple statistical methods (like Z-score or IQR), algorithmic detectors learn patterns
from your data and can detect more complex types of anomalies.

## How It Works

The typical workflow is:

1. Create a detector with your desired parameters
2. Call `Matrix{` to train the detector on "normal" data
3. Call `Matrix{` to identify outliers in new data
4. Optionally use `Matrix{` to get anomaly scores

## Properties

| Property | Summary |
|:-----|:--------|
| `IsFitted` | Gets a value indicating whether the detector has been fitted to data. |
| `Threshold` | Gets the threshold used to classify samples as inliers or outliers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DecisionFunction(Matrix<>)` | Computes the anomaly score for each sample. |
| `Fit(Matrix<>)` | Trains the outlier detector on the provided data. |
| `Predict(Matrix<>)` | Predicts whether each sample is an inlier or outlier. |

