---
title: "TimeSeriesModelType"
description: "Represents different types of time series forecasting models used for analyzing and predicting sequential data over time."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different types of time series forecasting models used for analyzing and predicting sequential data over time.

## How It Works

**For Beginners:** Time series models help us understand patterns in data that change over time and make predictions about future values.

Think of time series data as any measurement collected regularly over time - like daily temperature readings, 
monthly sales figures, or hourly website traffic. These models help us answer questions like:

- "What will our sales be next month?"
- "How many visitors will our website get tomorrow?"
- "What will the temperature be next week?"

Different models are designed to capture different patterns in time data:

- Some are good at finding seasonal patterns (like holiday shopping spikes)
- Others excel at detecting long-term trends (like gradual population growth)
- Some can handle sudden changes or outliers (like a viral social media post)

The right model depends on your specific data and what patterns you expect to find in it.

## Fields

| Field | Summary |
|:-----|:--------|
| `ARIMA` | Auto-Regressive Integrated Moving Average model - a standard statistical method for time series forecasting that combines autoregression, differencing, and moving average components. |
| `ARIMAX` | ARIMA model with additional explanatory variables (exogenous variables) that can influence the forecast. |
| `ARMA` | Auto-Regressive Moving Average model - combines autoregressive and moving average components without the differencing (integration) step. |
| `AutoRegressive` | Auto-Regressive model - predicts future values based solely on past values of the same variable. |
| `BayesianStructuralTimeSeriesModel` | A flexible Bayesian approach to time series modeling that incorporates prior knowledge and uncertainty. |
| `Custom` | Represents a custom or user-defined time series model not covered by the standard types. |
| `DoubleExponentialSmoothing` | An extension of simple exponential smoothing that can handle data with a trend component. |
| `DynamicRegressionWithARIMAErrors` | A model that combines regression with ARIMA modeling of the error terms to account for both external factors and time dependencies. |
| `ExponentialSmoothing` | A general class of forecasting methods that give more weight to recent observations and less weight to older observations. |
| `GARCH` | Generalized Autoregressive Conditional Heteroskedasticity model - specialized for forecasting volatility in time series. |
| `InterventionAnalysis` | Analyzes how specific events or interventions affect a time series and quantifies their impact. |
| `MA` | Moving Average model - predicts future values based on past forecast errors rather than past values. |
| `NeuralNetworkARIMA` | A hybrid model that combines neural networks with traditional ARIMA models to leverage the strengths of both approaches. |
| `ProphetModel` | A forecasting model developed by Facebook that handles multiple seasonality patterns and is robust to missing data and outliers. |
| `SARIMA` | Seasonal Auto-Regressive Integrated Moving Average model - extends ARIMA to handle data with seasonal patterns. |
| `STLDecomposition` | Seasonal and Trend decomposition using Loess - breaks down time series into trend, seasonal, and remainder components. |
| `SimpleExponentialSmoothing` | The most basic form of exponential smoothing that handles data with no clear trend or seasonality. |
| `SpectralAnalysis` | Analyzes time series data by decomposing it into different frequency components to identify cyclical patterns. |
| `StateSpace` | A flexible framework for time series modeling that represents a system's behavior using state variables. |
| `TBATS` | A flexible time series model that handles complex seasonal patterns using trigonometric components. |
| `TransferFunctionModel` | Models how one time series affects another with potential time delays between cause and effect. |
| `TripleExponentialSmoothing` | An extension of double exponential smoothing that can handle data with both trend and seasonal components. |
| `UnobservedComponentsModel` | Models time series by representing them as combinations of unobserved components like trend, cycle, and seasonality. |
| `VAR` | Vector Autoregression model - extends autoregressive models to multiple related time series that influence each other. |
| `VARMA` | Vector Autoregression Moving-Average model - combines VAR and moving average components for multiple related time series. |

