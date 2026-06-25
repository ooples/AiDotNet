---
title: "CCDM<T>"
description: "CCDM — Conditional Continuous Diffusion Model for Time Series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

CCDM — Conditional Continuous Diffusion Model for Time Series.

## For Beginners

CCDM generates future time series values using a diffusion
process, similar to how image generators create pictures by gradually refining random
noise. Instead of predicting a single future value, it produces a range of probable
outcomes, giving you confidence intervals for your forecasts. This is especially
useful in finance where understanding uncertainty is as important as the prediction itself.

## How It Works

CCDM extends continuous diffusion models for conditional time series generation,
operating in continuous space with a score-matching objective for high-quality
probabilistic forecasting.

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardForTraining(Tensor<>)` | Tape-aware training forward. |
| `ForwardNative(Tensor<>)` | Continuous diffusion reverse process using annealed Langevin dynamics. |
| `RequireTapeCompatibleLossFunction(ILossFunction<>)` | Enforces the constructor contract that a user-supplied `ILossFunction` must derive from `LossFunctionBase` for CCDM's tape-based training to work. |

