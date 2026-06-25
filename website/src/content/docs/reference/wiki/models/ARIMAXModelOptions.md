---
title: "ARIMAXModelOptions<T>"
description: "Configuration options for the ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) time series forecasting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) time series forecasting model.

## For Beginners

ARIMAX is like ARIMA (which predicts future values based on past patterns) but with an 
added superpower: it can also consider outside factors that might affect your data. For example, when predicting 
ice cream sales, ARIMA would only look at past sales patterns, but ARIMAX could also consider temperature data. 
Think of it as a weather forecaster who not only looks at past weather patterns but also considers upcoming events 
like a hurricane forming in the ocean that will likely affect the forecast.

## How It Works

ARIMAX extends the ARIMA model by incorporating external (exogenous) variables that may influence the time series.
This allows the model to account for known external factors when making predictions.

## Properties

| Property | Summary |
|:-----|:--------|
| `AROrder` | Gets or sets the order of the AutoRegressive (AR) component. |
| `DecompositionType` | Gets or sets the matrix decomposition method used for solving the model's equations. |
| `DifferenceOrder` | Gets or sets the order of differencing (Integration). |
| `ExogenousVariables` | Gets or sets the number of exogenous (external) variables to include in the model. |
| `MAOrder` | Gets or sets the order of the Moving Average (MA) component. |

