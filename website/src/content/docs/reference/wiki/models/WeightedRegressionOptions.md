---
title: "WeightedRegressionOptions<T>"
description: "Configuration options for weighted regression models, which assign different importance to different observations."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for weighted regression models, which assign different importance to different observations.

## For Beginners

Weighted regression lets you give some data points more influence than others.

In standard regression:

- All observations have equal influence on the model
- This assumes all data points are equally important and reliable

Weighted regression solves this by:

- Allowing you to assign different weights to different observations
- Giving more influence to observations with higher weights
- Reducing the impact of less reliable or less important data points

This approach is useful when:

- Some observations are more reliable than others
- Recent data is more relevant than older data
- Certain observations are known to be outliers
- Data points have different levels of measurement precision

For example, in time series forecasting, you might assign higher weights to
recent observations to make your model more responsive to recent trends.

This class lets you configure how the weighted regression model is structured.

## How It Works

Weighted regression is an extension of standard regression techniques that allows different observations to have 
different levels of influence on the model. By assigning weights to each observation, you can control how much 
each data point contributes to the parameter estimation process. This approach is particularly useful when 
dealing with heteroscedasticity (non-constant variance in errors), outliers, or when some observations are known 
to be more reliable or important than others. Common applications include time series analysis where recent 
observations may be weighted more heavily than older ones, or situations where observations have different 
measurement precision. This class inherits from RegressionOptions and adds parameters specific to weighted 
regression.

## Properties

| Property | Summary |
|:-----|:--------|
| `Order` | Gets or sets the order of the regression model. |
| `Weights` | Gets or sets the weights assigned to each observation. |

