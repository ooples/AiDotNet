---
title: "NLinearOptions<T>"
description: "Options for `NLinearModel` — the normalization-linear forecaster (Zeng et al., AAAI 2023)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Options for `NLinearModel` — the normalization-linear forecaster
(Zeng et al., AAAI 2023). It subtracts the last observed value from the window, applies a single linear
map, then adds the value back — a distribution-shift-robust baseline that, like DLinear, is a strong
modern control against heavier models.

## For Beginners

this model first "re-centers" the recent history around its most recent
point, learns a straight-line mapping to the future, then shifts the prediction back up by that same
recent value. The re-centering trick helps a lot when the overall level of the series drifts over time.

## How It Works

NLinear is the simplest member of the LTSF-Linear family: normalize the input window by subtracting
its last value, run one linear layer, then add the last value back to the output. That subtract/add
step makes the model robust to distribution shift between training and test (a common failure mode of
transformers on long-horizon forecasting), while keeping the model to a single linear map.

Typical sizes: a `LookbackWindow` of 24–96 steps with `ForecastHorizon` 1; the
parameter count is about `LookbackWindow × ForecastHorizon` — a few thousand weights at most.

**Reference:** Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. "Are Transformers Effective for
Time Series Forecasting?" AAAI 2023. https://arxiv.org/abs/2205.13504

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NLinearOptions` | Creates a new `NLinearOptions` with default values. |
| `NLinearOptions(NLinearOptions<>)` | Creates a deep copy of an existing `NLinearOptions`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Number of training samples processed per gradient update. |
| `Epochs` | Number of full passes over the training data. |
| `ForecastHorizon` | Forecast horizon. |
| `LearningRate` | Step size for gradient-descent optimization. |
| `LookbackWindow` | History length the linear map sees (input window). |

