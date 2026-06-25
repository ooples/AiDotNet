---
title: "TransferFunctionOptions<T, TInput, TOutput>"
description: "Configuration options for Transfer Function models, which model the dynamic relationship between input (exogenous) and output (endogenous) time series."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Transfer Function models, which model the dynamic relationship
between input (exogenous) and output (endogenous) time series.

## For Beginners

Transfer Function models help you understand how one time series affects another over time.

When analyzing relationships between time series:

- Simple regression assumes immediate effects (X affects Y at the same time point)
- But in reality, effects often occur with delays (X affects Y after several time periods)
- The effect might also be distributed over multiple time periods

Transfer Function models solve this by:

- Capturing how input variables affect output variables over time
- Modeling both immediate and delayed effects
- Accounting for the autocorrelation in the output series itself
- Combining elements of regression and time series analysis

This approach is useful for:

- Understanding how marketing campaigns affect sales over time
- Modeling how temperature changes affect energy consumption
- Analyzing how policy changes impact economic indicators

This class lets you configure the structure of the Transfer Function model.

## How It Works

Transfer Function models extend ARIMA (AutoRegressive Integrated Moving Average) models by incorporating 
the effects of one or more input (exogenous) time series on an output (endogenous) time series. They are 
particularly useful for modeling systems where there is a known causal relationship between input and output 
variables, with possible lagged effects. The transfer function component captures how changes in the input 
series affect the output series over time, while the ARIMA component models the autocorrelation structure 
of the output series. These models are widely used in fields such as economics, engineering, and environmental 
science for applications like dynamic system modeling, intervention analysis, and forecasting with leading 
indicators. This class provides configuration options for controlling the order of various components in the 
Transfer Function model.

## Properties

| Property | Summary |
|:-----|:--------|
| `AROrder` | Gets or sets the order of the AutoRegressive (AR) component. |
| `InputLagOrder` | Gets or sets the order of the input lag in the transfer function. |
| `MAOrder` | Gets or sets the order of the Moving Average (MA) component. |
| `Optimizer` | Gets or sets the optimizer used for parameter estimation. |
| `OutputLagOrder` | Gets or sets the order of the output lag in the transfer function. |

