---
title: "ForecastingModelBase<T>"
description: "Base class for financial forecasting models, adding forecasting-specific behavior on top of the core financial model infrastructure."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Finance.Base`

Base class for financial forecasting models, adding forecasting-specific behavior
on top of the core financial model infrastructure.

## For Beginners

Think of this as the "forecasting toolkit" that all time series
models share. It defines what every forecasting model must expose so the rest of the
library can treat them consistently.

## How It Works

This base class layers forecasting-specific requirements (like multi-step prediction
and instance normalization) on top of the shared financial model base.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ForecastingModelBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new forecasting model with deferred configuration. |
| `ForecastingModelBase(NeuralNetworkArchitecture<>,Int32,Int32,Int32,ILossFunction<>)` | Initializes a new forecasting model in native mode. |
| `ForecastingModelBase(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32)` | Initializes a new forecasting model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `PatchSize` |  |
| `Stride` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Combines multiple prediction chunks into a single long forecast tensor. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts the input window forward by replacing the oldest steps with predictions. |

