---
title: "TimeSeriesFoundationModelTask"
description: "Defines the tasks that a time series foundation model can perform."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the tasks that a time series foundation model can perform.

## For Beginners

Traditional time series models are built for a single purpose
(e.g., only forecasting). Foundation models are more flexible — the same model can
forecast future values, detect anomalies, classify patterns, fill in missing data,
or produce embeddings. This enum lets you tell the model which task you want it to perform.

## How It Works

Time series foundation models are versatile architectures that can handle multiple
downstream tasks beyond simple forecasting. This enum provides type-safe task selection
instead of error-prone string-based approaches.

## Fields

| Field | Summary |
|:-----|:--------|
| `AnomalyDetection` | Detect unusual patterns or outliers in a time series. |
| `Classification` | Classify an entire time series into one of several categories. |
| `Embedding` | Generate a fixed-size vector representation (embedding) of a time series. |
| `Forecasting` | Predict future values of a time series given historical context. |
| `Imputation` | Fill in missing values within a time series. |

