---
title: "GARCHModelOptions<T>"
description: "Configuration options for the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model, which is used for analyzing and forecasting volatility in time series data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model,
which is used for analyzing and forecasting volatility in time series data.

## For Beginners

Think of GARCH as a specialized tool for predicting how much something will
fluctuate or vary in the future, rather than predicting its exact value. It's particularly useful for
financial data like stock prices, where you might want to know not just whether a price will go up or down,
but how stable or volatile it will be. For example, GARCH can help predict whether tomorrow's stock price
will likely stay close to today's price or might swing dramatically in either direction. This is valuable
for risk management and option pricing in finance. The model works by recognizing that periods of high
volatility often cluster together (if today is volatile, tomorrow is likely to be volatile too).

## How It Works

GARCH models are specialized time series models designed to capture volatility clustering in financial
and economic data. Unlike standard time series models that focus on predicting the mean value, GARCH
models specifically model how the variance (volatility) of a time series changes over time, accounting
for periods of high and low volatility.

## Properties

| Property | Summary |
|:-----|:--------|
| `ARCHOrder` | Gets or sets the ARCH order (p), which determines how many past squared errors are used to model current volatility. |
| `GARCHOrder` | Gets or sets the GARCH order (q), which determines how many past volatility values are used to model current volatility. |
| `MaxIterations` | Gets or sets the maximum number of iterations allowed during parameter estimation. |
| `MeanModel` | Gets or sets the model used to predict the mean of the time series before modeling volatility. |
| `Tolerance` | Gets or sets the convergence tolerance for parameter estimation. |

