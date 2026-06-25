---
title: "TSDiff<T>"
description: "TSDiff — Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TSDiff — Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting.

## For Beginners

TSDiff generates probabilistic forecasts using a three-step
process: predict, refine, and synthesize. It first learns general time series patterns
through diffusion (gradually adding and removing noise), then refines predictions using
the model's own internal guidance. This self-guided approach produces high-quality
forecasts with well-calibrated uncertainty estimates.

## How It Works

TSDiff uses unconditional denoising diffusion as a self-supervised pretraining objective
with self-guided refinement for high-quality probabilistic forecasting.

**Reference:** Kollovieh et al., "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting", NeurIPS 2023.

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardForTraining(Tensor<>)` | Tape-aware training forward. |
| `ForwardNative(Tensor<>)` | DDPM reverse process with self-guided diffusion refinement. |
| `PadOrTruncateRank2(Tensor<>,Int32)` | Pads-or-truncates a rank-1 / rank-2 [1, len] tensor along the last axis to a target width via Engine ops (TensorNarrow for truncate; TensorConcatenate with a zero pad for extend). |

