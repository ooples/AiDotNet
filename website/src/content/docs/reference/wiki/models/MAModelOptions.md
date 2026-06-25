---
title: "MAModelOptions<T>"
description: "Configuration options for Moving Average (MA) models, which are used to analyze time series data by modeling the error terms as a linear combination of previous error terms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Moving Average (MA) models, which are used to analyze time series data
by modeling the error terms as a linear combination of previous error terms.

## For Beginners

A Moving Average (MA) model helps predict future values in a time series
(like daily temperatures, stock prices, or website traffic) based on recent unexpected changes or "surprises."

Imagine you're trying to predict tomorrow's temperature:

- An MA model doesn't just look at yesterday's actual temperature
- Instead, it focuses on the recent "surprises" - the differences between what was predicted and what actually happened
- It assumes that these recent surprises contain useful information about what might happen next

For example:

- If the weather has been consistently 2 degrees warmer than predicted for the past few days
- An MA model would adjust tomorrow's forecast to account for this pattern of surprises

This approach is particularly useful for data that has short-term patterns or fluctuations.
This class allows you to configure how the MA model will be built and optimized.

## How It Works

The Moving Average (MA) model is a fundamental component in time series analysis and forecasting.
Unlike autoregressive (AR) models that express the current value as a function of past values,
MA models express the current value as a function of past forecast errors (also called shocks or innovations).
This makes MA models particularly effective at capturing short-term, irregular patterns in time series data.
The model is defined by its order (q), which determines how many past error terms are included in the model.

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets or sets the learning rate that controls the step size in each iteration of the optimization process. |
| `MAOrder` | Gets or sets the order of the Moving Average model, which determines how many past error terms are included. |
| `MaxIterations` | Gets or sets the maximum number of iterations allowed for the optimization algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance that determines when the optimization algorithm should stop. |

