---
title: "TiDEOptions<T>"
description: "Options for `TiDEModel` — Time-series Dense Encoder (Das et al., \"Long-term Forecasting with TiDE: Time-series Dense Encoder\", TMLR 2023)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Options for `TiDEModel` — Time-series Dense Encoder (Das et al.,
"Long-term Forecasting with TiDE: Time-series Dense Encoder", TMLR 2023). A pure-MLP encoder/decoder
with a linear residual; on long-horizon benchmarks it matches or beats transformers at a fraction of
the cost. Faithful core: a ReLU encoder MLP + decoder projection + a linear skip from the input window.

## For Beginners

this model squeezes the recent history through a small neural network
("encoder") to a compact summary, expands that summary into the forecast ("decoder"), and also adds a
simple straight-line shortcut from the input so it never does worse than a linear model. The
`HiddenSize` knob sets how big that compact summary is.

## How It Works

TiDE encodes the lookback window with a multi-layer-perceptron (MLP) encoder into a latent vector of
size `HiddenSize`, decodes that to the forecast horizon, and adds a linear residual mapped
directly from the input window. Being an MLP, it captures non-linear structure that the purely-linear
DLinear/NLinear cannot, while remaining far cheaper than attention-based models.

Typical sizes: `LookbackWindow` 24–96, `HiddenSize` 32–256 (64 is a good
default), `ForecastHorizon` 1 for the supervised harness.

**Reference:** Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan Mathur, Rajat Sen, Rose Yu.
"Long-term Forecasting with TiDE: Time-series Dense Encoder." TMLR 2023. https://arxiv.org/abs/2304.08424

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TiDEOptions` | Creates a new `TiDEOptions` with default values. |
| `TiDEOptions(TiDEOptions<>)` | Creates a deep copy of an existing `TiDEOptions`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Number of training samples processed per gradient update. |
| `Epochs` | Number of full passes over the training data. |
| `ForecastHorizon` | Forecast horizon. |
| `HiddenSize` | Latent dimension of the MLP encoder (and decoder hidden width). |
| `LearningRate` | Step size for gradient-descent optimization. |
| `LookbackWindow` | History length the encoder sees (input window). |

