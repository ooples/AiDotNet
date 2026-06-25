---
title: "UnobservedComponentsOptions<T, TInput, TOutput>"
description: "Configuration options for Unobserved Components Models (UCM), which decompose time series into trend, seasonal, cycle, and irregular components."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Unobserved Components Models (UCM), which decompose time series into
trend, seasonal, cycle, and irregular components.

## For Beginners

Unobserved Components Models help you break down time series data into meaningful parts.

When analyzing time series data:

- It's often useful to separate the data into different components
- These components aren't directly observable but can be estimated

UCM decomposes time series into:

- Trend: The long-term direction (upward, downward, or stable)
- Seasonal: Regular patterns that repeat at fixed intervals (daily, weekly, monthly, etc.)
- Cycle: Irregular fluctuations that don't have a fixed period
- Irregular: Random noise or unexplained variation

This approach offers several benefits:

- Better understanding of what drives the time series
- Improved forecasting by modeling each component separately
- Ability to detect structural changes over time
- Flexibility to include or exclude components based on domain knowledge

This class lets you configure which components to include and how to estimate them.

## How It Works

Unobserved Components Models (UCM), also known as Structural Time Series Models, provide a flexible 
framework for decomposing time series data into distinct components that are not directly observable. 
These components typically include trend (long-term movement), seasonal (regular patterns at fixed 
intervals), cycle (irregular fluctuations of varying length), and irregular (random noise) components. 
UCM is particularly useful for understanding the underlying structure of time series data, forecasting, 
and detecting structural changes. This approach is based on state space models and is often estimated 
using the Kalman filter. This class provides configuration options for controlling the components 
included in the model and the estimation process.

## Properties

| Property | Summary |
|:-----|:--------|
| `CycleLambda` | Gets or sets the smoothing parameter for the cycle component. |
| `CycleMaxPeriod` | Gets or sets the maximum period for the cycle component. |
| `CycleMinPeriod` | Gets or sets the minimum period for the cycle component. |
| `Decomposition` | Gets or sets the matrix decomposition method used in the estimation algorithm. |
| `IncludeCycle` | Gets or sets a value indicating whether to include a cycle component in the model. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the estimation algorithm. |
| `OptimizeParameters` | Gets or sets a value indicating whether to optimize the model parameters. |
| `Optimizer` | Gets or sets the optimizer used for parameter estimation. |
| `SeasonalPeriod` | Gets or sets the seasonal period for the model. |

