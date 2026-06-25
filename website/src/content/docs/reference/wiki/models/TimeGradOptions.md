---
title: "TimeGradOptions<T>"
description: "Configuration options for TimeGrad (Autoregressive Denoising Diffusion Model for Time Series)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TimeGrad (Autoregressive Denoising Diffusion Model for Time Series).

## For Beginners

TimeGrad brings the power of diffusion models (like those
used in image generation) to time series forecasting:

**The Key Insight:**
Most forecasting models give you ONE prediction. But in practice, you want to know
"how uncertain is this prediction?" TimeGrad solves this by modeling the FULL probability
distribution of future values using a diffusion process.

**How Diffusion Works (simplified):**

1. **Forward Process:** Gradually add noise to data until it becomes pure noise
2. **Reverse Process:** Learn to remove noise step-by-step, generating samples
3. **Conditioning:** Use historical data to guide the denoising
4. **Sampling:** Generate multiple forecasts from the learned distribution

**TimeGrad Architecture:**

- RNN encoder processes historical data
- Diffusion model generates future values conditioned on hidden state
- Multiple samples give uncertainty estimates

**Key Benefits:**

- Probabilistic forecasts (not just point predictions)
- Well-calibrated uncertainty estimates
- Can generate diverse forecast scenarios
- State-of-the-art accuracy on probabilistic metrics

## How It Works

TimeGrad is a probabilistic time series forecasting model that uses denoising diffusion
to generate accurate forecasts with well-calibrated uncertainty estimates.

**Reference:** Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", 2021.
https://arxiv.org/abs/2101.12072

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeGradOptions` | Initializes a new instance of the `TimeGradOptions` class with default values. |
| `TimeGradOptions(TimeGradOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Gets or sets the ending noise level (beta_T). |
| `BetaSchedule` | Gets or sets the noise schedule type. |
| `BetaStart` | Gets or sets the starting noise level (beta_1). |
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DenoisingNetworkDim` | Gets or sets the dimension of the denoising network. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension for the RNN encoder. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion steps (T in the paper). |
| `NumRnnLayers` | Gets or sets the number of RNN layers in the encoder. |
| `NumSamples` | Gets or sets the number of samples to generate for probabilistic forecasting. |
| `UseResidualConnection` | Gets or sets whether to use residual connections in the denoising network. |

