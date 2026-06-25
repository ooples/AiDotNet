---
title: "TimeDiffOptions<T>"
description: "Configuration options for TimeDiff (Non-autoregressive Diffusion-based Time Series Forecasting)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TimeDiff (Non-autoregressive Diffusion-based Time Series Forecasting).

## How It Works

TimeDiff extends DDPM with novel conditioning mechanisms specifically designed for
time series: future-mixup for training, autoregressive initialization for inference,
and a transformer-based denoiser.

**Reference:** Shen & Kwok, "Non-autoregressive Conditional Diffusion Models for Time Series Prediction", ICML 2023.

## Properties

| Property | Summary |
|:-----|:--------|
| `UseAutoregressiveInit` | Gets or sets whether to use autoregressive initialization at inference. |
| `UseFutureMixup` | Gets or sets whether to use future-mixup augmentation during training. |

