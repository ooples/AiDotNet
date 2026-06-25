---
title: "GIFTEvalBenchmark<T>"
description: "GIFT-Eval benchmark implementation for standardized evaluation of time series foundation models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Evaluation`

GIFT-Eval benchmark implementation for standardized evaluation of time series foundation models.

## For Beginners

GIFT-Eval (General Time Series Forecasting Model Evaluation) is the
standard benchmark used by the research community to compare foundation models. It uses two
key metrics:

- **MASE** (Mean Absolute Scaled Error): Measures point forecast accuracy relative

to a naive baseline. A MASE of 1.0 means the model performs the same as naive forecasting;
values below 1.0 mean it's better.

- **CRPS** (Continuous Ranked Probability Score): Measures probabilistic forecast

quality by comparing predicted distributions to actual values. Lower is better.

## How It Works

**Reference:** Jain et al., "GIFT-Eval: A Benchmark for General Time Series Forecasting
Model Evaluation", 2024. https://arxiv.org/abs/2410.10393

**Leaderboard:** https://huggingface.co/spaces/Salesforce/GIFT-Eval

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCRPS(IReadOnlyDictionary<Double,Tensor<>>,Tensor<>)` | Computes the Continuous Ranked Probability Score (CRPS) for quantile/probabilistic forecasts. |
| `ComputeMASE(Tensor<>,Tensor<>,Tensor<>,Int32)` | Computes the Mean Absolute Scaled Error (MASE) for point forecasts. |
| `ComputeQuantileLoss(Tensor<>,Tensor<>,Double)` | Computes the quantile loss (pinball loss) for a specific quantile level. |
| `RunBenchmark(ITimeSeriesFoundationModel<>,IReadOnlyList<Tensor<>>,IReadOnlyList<Int32>,IReadOnlyList<Int32>,Double[])` | Runs a comprehensive GIFT-Eval style evaluation on a foundation model. |

