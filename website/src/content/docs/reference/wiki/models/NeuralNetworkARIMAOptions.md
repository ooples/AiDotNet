---
title: "NeuralNetworkARIMAOptions<T>"
description: "Configuration options for Neural Network ARIMA (AutoRegressive Integrated Moving Average) models,  which combine traditional statistical time series methods with neural networks for improved forecasting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Neural Network ARIMA (AutoRegressive Integrated Moving Average) models, 
which combine traditional statistical time series methods with neural networks for improved forecasting.

## For Beginners

Neural Network ARIMA combines two powerful approaches for predicting future values in a time series:

Imagine you're trying to predict tomorrow's temperature:

- ARIMA is like using mathematical formulas that look at recent temperatures and patterns
- Neural Networks are like having a smart system that can learn complex relationships from data
- Neural Network ARIMA combines both approaches to get better predictions

Traditional ARIMA is good at capturing:

- How today's temperature relates to yesterday's (AR - AutoRegressive)
- How random fluctuations from previous days affect today (MA - Moving Average)

But it struggles with complex patterns like:

- When a cold front arrives, temperatures might drop suddenly but then recover differently than normal
- How humidity and cloud cover might interact in non-obvious ways to affect temperature

The neural network part helps capture these complex relationships, while the ARIMA part
ensures the basic time patterns are properly handled. This combination often produces
better forecasts than either approach alone.

This class lets you configure both the ARIMA components and the neural network that will
work together to make predictions.

## How It Works

Neural Network ARIMA is a hybrid approach that extends traditional ARIMA models by incorporating
neural networks to capture complex nonlinear patterns in time series data. This approach leverages
both the statistical foundation of ARIMA for handling linear dependencies, seasonality, and trends,
while using neural networks to model nonlinear relationships that traditional ARIMA cannot capture.
The resulting model can provide more accurate forecasts for time series with complex patterns,
regime shifts, or other nonlinear dynamics that are common in real-world data.

## Properties

| Property | Summary |
|:-----|:--------|
| `AROrder` | Gets or sets the AutoRegressive (AR) order, which determines how many previous time steps are used as inputs to predict the current value. |
| `DifferencingOrder` | The order of differencing applied to the time series data. |
| `ExogenousVariables` | Gets or sets the number of exogenous (external) variables to include in the model. |
| `LaggedPredictions` | Gets or sets the number of lagged predictions to use as inputs to the neural network. |
| `MAOrder` | Gets or sets the Moving Average (MA) order, which determines how many previous error terms are used in the prediction model. |
| `NeuralNetwork` | Gets or sets the neural network to use in the hybrid model. |
| `OptimizeParameters` | Gets or sets a value indicating whether the model should attempt to optimize its parameters during training. |
| `Optimizer` | Gets or sets the optimizer to use for training the neural network component. |

