---
title: "MatrixProfileDetector<T>"
description: "Detects anomalies using Matrix Profile for time series discord detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TimeSeries`

Detects anomalies using Matrix Profile for time series discord detection.

## For Beginners

Matrix Profile computes the distance to the nearest neighbor subsequence
for every subsequence in a time series. Subsequences with no similar matches (discords)
are anomalies. It's one of the most powerful time series anomaly detection methods.

## How It Works

The algorithm works by:

1. Extract all subsequences of length m
2. For each subsequence, find the nearest neighbor (excluding trivial matches)
3. Store these distances as the Matrix Profile
4. High values in the Matrix Profile indicate discords (anomalies)

**When to use:**

- Time series pattern anomaly detection
- Finding unusual subsequences
- Discord discovery in time series

**Industry Standard Defaults:**

- Subsequence length: 50 (or approximately one period)
- Exclusion zone: m/4
- Contamination: 0.1 (10%)

Reference: Yeh, C.C.M., et al. (2016). "Matrix Profile I: All Pairs Similarity Joins for
Time Series: A Unifying View." IEEE ICDM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MatrixProfileDetector(Int32,Int32,Double,Int32)` | Creates a new Matrix Profile anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExclusionZone` | Gets the exclusion zone size. |
| `SubsequenceLength` | Gets the subsequence length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `ValueDeviationWeight` | Weight for the value-deviation component in the combined anomaly score. |

