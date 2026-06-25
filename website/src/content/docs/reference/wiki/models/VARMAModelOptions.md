---
title: "VARMAModelOptions<T>"
description: "Configuration options for Vector Autoregressive Moving Average (VARMA) models, which extend VAR models by incorporating moving average terms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Vector Autoregressive Moving Average (VARMA) models, which extend VAR models
by incorporating moving average terms.

## For Beginners

VARMA models extend VAR models by including both past values and past errors.

When modeling multiple related time series:

- VAR models use past values of all variables to predict future values
- VARMA models add another component: past prediction errors

This additional component:

- Captures patterns in the errors or "shocks" to the system
- Can lead to more accurate and efficient models
- Is particularly useful when the effects of shocks persist for multiple periods

Think of it like this:

- VAR: "Tomorrow's values depend on today's and yesterday's values"
- VARMA: "Tomorrow's values depend on today's and yesterday's values, plus how wrong our recent predictions were"

This class lets you configure the moving average component of VARMA models,
while inheriting all the configuration options for VAR models.

## How It Works

Vector Autoregressive Moving Average (VARMA) models extend Vector Autoregressive (VAR) models by incorporating 
moving average (MA) terms. While VAR models express each variable as a linear function of past values of itself 
and past values of other variables, VARMA models also include past error terms. This additional flexibility can 
lead to more parsimonious models and better forecasting performance, especially when the true data generating 
process includes moving average components. VARMA models are particularly useful for modeling and forecasting 
multiple interrelated time series, capturing both the autoregressive and moving average dynamics in the system. 
This class inherits from VARModelOptions and adds parameters specific to the moving average component of VARMA 
models.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaLag` | Gets or sets the lag order for the Moving Average (MA) component. |

