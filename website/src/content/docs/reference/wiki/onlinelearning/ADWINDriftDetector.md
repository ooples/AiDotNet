---
title: "ADWINDriftDetector<T>"
description: "ADWIN (ADaptive WINdowing) drift detector for concept drift detection in data streams."
section: "API Reference"
---

`Models & Types` · `AiDotNet.OnlineLearning`

ADWIN (ADaptive WINdowing) drift detector for concept drift detection in data streams.

## For Beginners

ADWIN is like a smart sliding window that automatically
adjusts its size based on whether the data is stable or changing:

When data is stable:

- Window grows to include more history
- More data = more accurate estimates

When drift occurs:

- Window shrinks to forget old (now irrelevant) data
- Model adapts quickly to new patterns

How it works:

1. Maintains a window W of recent values
2. For each new value, checks if there's a "cut point" where:
- W₁ = data before cut, W₂ = data after cut
- If |mean(W₁) - mean(W₂)| > threshold, drift is detected
3. When drift detected, discards W₁ (old data)

The threshold is based on the Hoeffding bound, providing statistical guarantees:

- P(false alarm) bounded by δ
- P(missing real drift) bounded by δ

Key advantage: No need to set window size manually - ADWIN adapts!

Usage:

References:

- Bifet & Gavaldà (2007). "Learning from Time-Changing Data with Adaptive Windowing"

## How It Works

ADWIN maintains a variable-length window of recent data and automatically shrinks it
when a significant change in the mean is detected. It provides theoretical guarantees
on false positive/negative rates.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ADWINDriftDetector(Double,Int32)` | Initializes a new instance of the ADWINDriftDetector class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsDriftDetected` | Gets whether drift has been detected. |
| `IsWarning` | Gets whether a warning has been detected. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompressBuckets` | Compresses buckets using exponential histogram scheme. |
| `ComputeADWINBound(Int64,Int64)` | Computes the ADWIN bound for cut detection. |
| `DetectDrift` | Detects drift using ADWIN's cut-based test. |
| `GetChangePoint` | Gets the estimated change point. |
| `GetStatistics` | Gets current detection statistics. |
| `GetWindowMean` | Gets the current mean of the window. |
| `GetWindowSize` | Gets the current window size. |
| `GetWindowVariance` | Gets the estimation of variance in the window. |
| `InsertElement(Double)` | Inserts a new element into the bucket list. |
| `MergeBuckets(Int32,Int32)` | Merges two buckets into the first one. |
| `Reset` | Resets the detector to its initial state. |
| `Update()` | Updates the detector with a new observation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bucketList` | The bucket list representing the adaptive window. |
| `_changePoint` | Index of the last detected change point. |
| `_delta` | The confidence parameter (probability bound for false positives/negatives). |
| `_driftDetected` | Whether drift was detected on the last update. |
| `_maxBucketsPerLevel` | Maximum number of buckets per level. |
| `_numOps` | Numeric operations helper for generic math. |
| `_sampleCount` | Count of total samples processed. |
| `_total` | Total sum of values in the window. |
| `_variance` | Total variance (sum of squared deviations) in the window. |
| `_width` | Total count of values in the window. |

