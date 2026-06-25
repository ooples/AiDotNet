---
title: "VARModelOptions<T>"
description: "Configuration options for Vector Autoregressive (VAR) models, which model the linear interdependencies among multiple time series."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Vector Autoregressive (VAR) models, which model the linear interdependencies
among multiple time series.

## For Beginners

VAR models help you understand and forecast multiple related time series simultaneously.

When dealing with multiple time series that affect each other:

- Simple models treat each series independently
- But in reality, variables often influence each other over time

VAR models solve this by:

- Modeling all variables together as a system
- Allowing each variable to depend on past values of all variables
- Capturing the relationships and feedback effects between variables

This approach offers several benefits:

- Better forecasts by incorporating relationships between variables
- Understanding how shocks to one variable affect other variables
- Analyzing the dynamic interactions in a system

For example, in economics, VAR models might show how changes in interest rates
affect GDP, inflation, and unemployment over several quarters.

This class lets you configure how the VAR model is structured and estimated.

## How It Works

Vector Autoregressive (VAR) models are a generalization of univariate autoregressive models to multivariate 
time series. In a VAR model, each variable is modeled as a linear function of past values of itself and past 
values of all other variables in the system. This approach captures the dynamic relationships and feedback 
effects among multiple interrelated time series. VAR models are widely used in economics, finance, and other 
fields for forecasting, structural analysis, and policy analysis. They provide a flexible framework for 
analyzing the joint dynamics of multiple variables without imposing strong a priori restrictions on the 
relationships. This class provides configuration options for controlling the structure and estimation of 
VAR models.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the type of matrix decomposition used in the estimation algorithm. |
| `Lag` | Gets or sets the lag order for the VAR model. |
| `OutputDimension` | Gets or sets the dimension of the output vector. |

