---
title: "BayesianStructuralTimeSeriesOptions<T>"
description: "Configuration options for Bayesian Structural Time Series models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Bayesian Structural Time Series models.

## For Beginners

A time series is simply data collected over time (like daily temperatures, 
monthly sales, or yearly population). Bayesian Structural Time Series is a powerful way to analyze this 
kind of data by breaking it down into different parts: the overall direction (trend), repeating patterns 
(seasonality), and the influence of other factors (regression). The "Bayesian" part means it can express 
uncertainty about its predictions and incorporate what you already know about the data. Think of it like 
weather forecasting that not only predicts tomorrow's temperature but also tells you how confident it is 
in that prediction.

## How It Works

Bayesian Structural Time Series (BSTS) models are flexible time series models that decompose a time series
into trend, seasonal, and regression components. They use Bayesian methods to estimate model parameters
and can handle missing data, incorporate prior knowledge, and provide uncertainty estimates.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvergenceTolerance` | Gets or sets the convergence tolerance for parameter estimation. |
| `IncludeRegression` | Gets or sets whether to include regression components in the model. |
| `InitialLevelValue` | Gets or sets the initial value for the level component of the time series. |
| `InitialObservationVariance` | Gets or sets the initial variance of the observation noise. |
| `InitialTrendValue` | Gets or sets the initial value for the trend component of the time series. |
| `LevelSmoothingPrior` | Gets or sets the prior for the level component's smoothing parameter. |
| `MaxIterations` | Gets or sets the maximum number of iterations for parameter estimation. |
| `PerformBackwardSmoothing` | Gets or sets whether to perform backward smoothing after forward filtering. |
| `RegressionDecompositionType` | Gets or sets the matrix decomposition method used for regression calculations. |
| `RidgeParameter` | Gets or sets the ridge parameter for regression regularization. |
| `SeasonalPeriods` | Gets or sets the list of seasonal periods to model in the time series. |
| `SeasonalSmoothingPrior` | Gets or sets the prior for the seasonal component's smoothing parameter. |
| `TrendSmoothingPrior` | Gets or sets the prior for the trend component's smoothing parameter. |
| `UseAutomaticPriors` | Gets or sets whether to automatically determine prior distributions. |

