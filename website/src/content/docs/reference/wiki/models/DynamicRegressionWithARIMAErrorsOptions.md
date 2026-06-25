---
title: "DynamicRegressionWithARIMAErrorsOptions<T>"
description: "Configuration options for Dynamic Regression with ARIMA Errors, a powerful time series forecasting method that combines regression with time series error correction."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Dynamic Regression with ARIMA Errors, a powerful time series forecasting method
that combines regression with time series error correction.

## For Beginners

This is a forecasting method that combines two powerful techniques. First, it uses
regression to find relationships between your target variable (what you're trying to predict) and other variables
that might influence it (like temperature affecting ice cream sales). Then, it analyzes the errors in that
prediction to find patterns over time (like seasonal trends or cycles). By combining both approaches, it often
produces more accurate forecasts than either method alone. Think of it as not just predicting based on related
factors, but also learning from its own mistakes.

## How It Works

Dynamic Regression with ARIMA Errors (sometimes called RegARIMA) is a hybrid forecasting approach that uses
external variables (regressors) to explain the main trend while modeling the residual errors with ARIMA
(AutoRegressive Integrated Moving Average) to capture temporal patterns not explained by the regressors.

## Properties

| Property | Summary |
|:-----|:--------|
| `AROrder` | Gets or sets the AutoRegressive (AR) order for the ARIMA component. |
| `DecompositionType` | Gets or sets the matrix decomposition method used for solving the regression equations. |
| `DifferenceOrder` | Gets or sets the differencing order for the ARIMA component. |
| `ExternalRegressors` | Gets or sets the number of external regressor variables to use in the model. |
| `MAOrder` | Gets or sets the Moving Average (MA) order for the ARIMA component. |
| `Regularization` | Gets or sets the regularization method used to prevent overfitting in the regression component. |

