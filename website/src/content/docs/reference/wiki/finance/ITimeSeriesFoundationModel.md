---
title: "ITimeSeriesFoundationModel<T>"
description: "Interface for multi-task time series foundation models that support forecasting, anomaly detection, classification, imputation, and embedding generation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

Interface for multi-task time series foundation models that support forecasting,
anomaly detection, classification, imputation, and embedding generation.

## For Beginners

Think of foundation models as "Swiss Army knives" for time series:

**Traditional approach:** Train separate models for each task

- Model A: Forecasting
- Model B: Anomaly detection
- Model C: Classification
- Model D: Imputation

**Foundation model approach:** One pretrained model, many tasks

- Same model handles forecasting, anomaly detection, classification, etc.
- Works out of the box on new data (zero-shot capability)
- Can be fine-tuned for better performance on specific datasets

**Key Benefits:**

- Reduced development time — no need to train multiple models
- Better generalization — pretrained on diverse data
- Consistent feature extraction — shared representations across tasks
- Lower total compute — one model instead of many

## How It Works

Time series foundation models are large pretrained models that can handle multiple
downstream tasks using a single architecture. Unlike traditional models that are
purpose-built for one task (e.g., only forecasting), foundation models learn general
representations of time series data during pretraining and can be applied to various
tasks with minimal or no fine-tuning.

**Reference:** This interface is inspired by models such as MOMENT (CMU, ICML 2024),
Chronos-2 (Amazon, 2025), Moirai 2.0 (Salesforce, 2025), TimesFM 2.5 (Google, 2025),
and Tiny Time Mixers (IBM, NeurIPS 2024).

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentTask` | Gets the task this model is currently configured to perform. |
| `MaxContextLength` | Gets the maximum context length (number of input time steps) the model can process. |
| `MaxPredictionHorizon` | Gets the maximum prediction horizon the model can produce in a single forward pass. |
| `ModelSize` | Gets the size variant of this foundation model. |
| `SupportedTasks` | Gets the list of tasks this foundation model supports. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Classify(Tensor<>,Int32)` | Classifies the input time series into one of several categories. |
| `DetectAnomalies(Tensor<>,Nullable<Double>)` | Detects anomalies in the input time series. |
| `Embed(Tensor<>)` | Generates a fixed-size embedding vector for the input time series. |
| `Impute(Tensor<>,Tensor<>)` | Fills in missing values in a time series using the surrounding context. |

