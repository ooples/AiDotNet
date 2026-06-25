---
title: "DLinearOptions<T>"
description: "Options for `DLinearModel` — the decomposition-linear forecaster (Zeng et al., \"Are Transformers Effective for Time Series Forecasting?\", AAAI 2023)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Options for `DLinearModel` — the decomposition-linear forecaster
(Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023). Despite its
simplicity it is a strong, current baseline that frequently matches or beats heavier transformers on
standard long-horizon benchmarks, at a fraction of the cost.

## For Beginners

this model splits a series into its slow-moving "trend" (the moving
average) and the leftover wiggles ("seasonal"), learns a simple straight-line mapping for each, and
adds them back together to predict the future. It is fast, hard to overfit, and surprisingly accurate,
which makes it an excellent first model to try and a yardstick for judging fancier ones.

## How It Works

DLinear decomposes each input window into a trend component (a moving average) and the remaining
seasonal component, applies one independent linear map to each, and sums the two projections to
produce the forecast. The whole model is two linear layers — there is no attention and no recurrence —
so it trains in seconds and is a natural sanity-check baseline before reaching for a transformer.

Typical sizes: a `LookbackWindow` of 24–96 steps with `ForecastHorizon` 1
(the supervised harness predicts the next value); the parameter count is roughly
`2 × LookbackWindow × ForecastHorizon`, i.e. a few thousand parameters even for long windows.

**Reference:** Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. "Are Transformers Effective for
Time Series Forecasting?" AAAI 2023. https://arxiv.org/abs/2205.13504

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DLinearOptions` | Creates a new `DLinearOptions` with default values. |
| `DLinearOptions(DLinearOptions<>)` | Creates a deep copy of an existing `DLinearOptions`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Number of training samples processed per gradient update. |
| `Epochs` | Number of full passes over the training data. |
| `ForecastHorizon` | Forecast horizon. |
| `LearningRate` | Step size for gradient-descent optimization. |
| `LookbackWindow` | History length the linear maps see (input window). |
| `MovingAverageKernel` | Moving-average kernel for the trend/seasonal decomposition (odd; clamped to the window). |

