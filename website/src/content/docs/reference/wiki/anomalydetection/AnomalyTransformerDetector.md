---
title: "AnomalyTransformerDetector<T>"
description: "Implements Anomaly Transformer for time series anomaly detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TimeSeries`

Implements Anomaly Transformer for time series anomaly detection.

## For Beginners

Anomaly Transformer uses the attention mechanism to detect anomalies.
Normal points tend to have similar attention patterns to their neighbors, while anomalies
show distinct attention patterns. It uses "Association Discrepancy" as the anomaly score.

## How It Works

The algorithm works by:

1. Encode time series using positional encoding
2. Apply self-attention to learn temporal relationships
3. Compute association discrepancy between prior and series associations
4. High discrepancy indicates anomalies

**When to use:**

- Long time series with complex patterns
- When you need to capture long-range dependencies
- Multivariate time series anomaly detection

**Industry Standard Defaults:**

- Model dimensions: 64
- Number of heads: 4
- Sequence length: 100
- Epochs: 10
- Contamination: 0.1 (10%)

Reference: Xu, J., et al. (2022).
"Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." ICLR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnomalyTransformerDetector(Int32,Int32,Int32,Int32,Double,Double,Int32)` | Creates a new Anomaly Transformer detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDim` | Gets the model dimensions. |
| `NumHeads` | Gets the number of attention heads. |
| `SeqLength` | Gets the sequence length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

