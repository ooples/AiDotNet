---
title: "SpectralResidualDetector<T>"
description: "Detects anomalies in time series using Spectral Residual method."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TimeSeries`

Detects anomalies in time series using Spectral Residual method.

## For Beginners

Spectral Residual is inspired by visual saliency detection in images.
It transforms the time series to the frequency domain, finds what's "unusual" in the
spectrum (the saliency), and transforms back to identify anomalies.

## How It Works

The algorithm works by:

1. Apply FFT to convert time series to frequency domain
2. Compute the log amplitude spectrum
3. Extract spectral residual by subtracting smoothed spectrum
4. Transform back to get saliency map (anomaly scores)

**When to use:**

- Time series with recurring patterns at multiple frequencies
- When anomalies disrupt the normal frequency pattern
- Large-scale time series (efficient O(n log n) complexity)

**Industry Standard Defaults:**

- Window size: 3 (for spectrum smoothing)
- Score threshold: Based on contamination
- Contamination: 0.1 (10%)

Reference: Microsoft's SR-CNN: Spectral Residual based anomaly detection.
Originally from Hou, X., Zhang, L. (2007). "Saliency Detection: A Spectral Residual Approach."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralResidualDetector(Int32,Double,Int32)` | Creates a new Spectral Residual anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `WindowSize` | Gets the window size for spectrum smoothing. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

