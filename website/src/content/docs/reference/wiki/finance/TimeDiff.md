---
title: "TimeDiff<T>"
description: "TimeDiff — Non-autoregressive Conditional Diffusion Models for Time Series Prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TimeDiff — Non-autoregressive Conditional Diffusion Models for Time Series Prediction.

## For Beginners

TimeDiff improves diffusion-based forecasting with two clever
tricks. During training, it mixes future values into the input (future-mixup) to help the
model learn what comes next. During prediction, it uses an initial rough forecast to guide
the diffusion process, producing all future values at once rather than one at a time, which
is both faster and more consistent.

## How It Works

TimeDiff extends DDPM with future-mixup training augmentation and autoregressive initialization
at inference for high-quality non-autoregressive time series forecasting.

**Reference:** Shen & Kwok, "Non-autoregressive Conditional Diffusion Models for Time Series Prediction", ICML 2023.

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardForTraining(Tensor<>)` | Tape-aware training forward. |
| `ForwardNative(Tensor<>)` | DDPM reverse process with optional autoregressive initialization. |
| `PadOrTruncateRank2(Tensor<>,Int32)` | Pads-or-truncates a rank-1 / rank-2 [1, len] tensor along the last axis to a target width via Engine ops. |

