---
title: "TimeSeriesFoundationModelBase<T>"
description: "Abstract base class for time series foundation models that support multiple downstream tasks."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Finance.Base`

Abstract base class for time series foundation models that support multiple downstream tasks.

## For Beginners

This is the foundation that all time series foundation models build upon.
It provides:

- Common infrastructure for both ONNX and native mode operation
- Default "not supported" implementations for optional tasks
- A `ValidateTaskSupported` helper to check task compatibility
- Standard properties for model size, parameter count, and context limits

Models that support only forecasting (like TimesFM) can inherit this class and only
override the forecasting-related methods. Multi-task models (like MOMENT) override
the additional task methods they support.

## How It Works

This base class extends `ForecastingModelBase` with multi-task capabilities
defined by `ITimeSeriesFoundationModel`. It provides default implementations
that throw `NotSupportedException` for optional tasks, allowing single-task models
(e.g., forecasting-only) to inherit without implementing every method.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesFoundationModelBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new foundation model with deferred configuration. |
| `TimeSeriesFoundationModelBase(NeuralNetworkArchitecture<>,Int32,Int32,Int32,ILossFunction<>)` | Initializes a new foundation model in native mode. |
| `TimeSeriesFoundationModelBase(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32)` | Initializes a new foundation model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentTask` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `SupportedTasks` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Classify(Tensor<>,Int32)` |  |
| `DetectAnomalies(Tensor<>,Nullable<Double>)` |  |
| `Embed(Tensor<>)` |  |
| `Impute(Tensor<>,Tensor<>)` |  |
| `ValidateTaskSupported(TimeSeriesFoundationModelTask)` | Validates that the specified task is supported by this model. |

