---
title: "MGTSD<T>"
description: "MG-TSD — Multi-Granularity Time Series Diffusion Model with Guided Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

MG-TSD — Multi-Granularity Time Series Diffusion Model with Guided Learning.

## For Beginners

MG-TSD forecasts at multiple zoom levels simultaneously.
It first makes a rough forecast (like predicting monthly trends), then uses that to
guide a more detailed forecast (like daily values). This coarse-to-fine approach is
similar to how an artist first sketches the broad outlines before adding fine details,
resulting in more coherent and accurate probabilistic predictions.

## How It Works

MG-TSD captures temporal patterns at multiple granularities using a coarse-to-fine
guidance mechanism where predictions at coarser levels guide fine-grained diffusion.

**Reference:** Fan et al., "MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process", ICLR 2024.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeInstanceStats(Tensor<>)` | Per-instance (RevIN) statistics over the input series: mean and (eps-stabilized) std. |
| `ForwardForTraining(Tensor<>)` | Tape-aware training forward: builds x_t from the stored (target, noise, timestep) and runs the denoising network as an x0-predictor over the same [x_t \| cond \| guidance \| t] pack used at inference. |
| `ForwardNative(Tensor<>)` | Multi-granularity DDPM reverse process with coarse-to-fine guided denoising. |
| `GetNamedLayerActivations(Tensor<>)` | Override the base class's naive linear-chain Layers walk because MGTSD's Layers list isn't a single pipeline — it's [inputProjection, denoisingLayers..., outputProjection] where each stage takes a different input shape (contextLength → hidd… |
| `GetOrCreateBaseOptimizer` | Tape-aware training forward. |
| `TryGetArchitectureInputShape` | MGTSD's `Layers` chain consumes the concatenated [x_t \| conditioning \| guidance \| t-embedding] pack of length forecastHorizon + hiddenDimension + forecastHorizon + 1 (24 + 128 + 24 + 1 = 177 on the default options), NOT Architecture.InputWi… |

