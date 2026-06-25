---
title: "CSDI<T>"
description: "CSDI — Conditional Score-based Diffusion Model for Probabilistic Time Series Imputation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

CSDI — Conditional Score-based Diffusion Model for Probabilistic Time Series Imputation.

## For Beginners

CSDI fills in missing data points in time series and forecasts
future values using a diffusion process. Think of it like an artist restoring a damaged
painting: it looks at the intact parts and intelligently fills in the gaps. Unlike simpler
methods that fill one gap at a time, CSDI fills all missing values simultaneously, which
produces more consistent and realistic results.

## How It Works

CSDI uses score-based diffusion for non-autoregressive time series imputation and forecasting.
It conditions on observed values using a transformer-based denoiser and generates all missing
values simultaneously.

**Reference:** Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", NeurIPS 2021.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDenoisingPairTape(Tensor<>,Tensor<>)` | Builds the (predicted-noise, true-noise) pair for one DDPM training step. |
| `ForwardNative(Tensor<>)` | DDPM reverse process: iteratively denoise from pure noise conditioned on observed values. |
| `GetOptions` |  |

