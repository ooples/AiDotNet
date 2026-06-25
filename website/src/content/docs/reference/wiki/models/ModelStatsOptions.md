---
title: "ModelStatsOptions"
description: "Configuration options for model statistics and diagnostics calculations, which help evaluate the quality, reliability, and performance of machine learning models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for model statistics and diagnostics calculations, which help evaluate
the quality, reliability, and performance of machine learning models.

## For Beginners

When building machine learning models, we need ways to check if they're
working well and to identify potential problems. This is similar to how a doctor uses various tests
to check your health.

Think of this class as a collection of settings for these "health checks" for your models:

- Some settings help detect when input variables are too similar (which can confuse models)
- Some settings configure how we evaluate recommendation or ranking systems
- Some settings control how we analyze patterns over time in time series data

By adjusting these settings, you can customize how thorough or sensitive these diagnostic
checks should be, similar to how medical tests can be adjusted for different levels of sensitivity.
The default values work well for most situations, but sometimes you'll want to adjust them based
on your specific data and model.

## How It Works

The ModelStatsOptions class provides configuration parameters for various statistical measures
and diagnostics that assess model quality. These include tests for multicollinearity (when features
are too closely related), metrics for ranking quality (MAP and NDCG), and tools for time series
analysis (ACF and PACF). Different diagnostic tools are appropriate for different types of models,
and this class allows customization of how these diagnostics are calculated.

## Properties

| Property | Summary |
|:-----|:--------|
| `AcfMaxLag` | Gets or sets the maximum lag to use for the ACF calculation. |
| `ConditionNumberMethod` | Gets or sets the method used to calculate the condition number, which measures how numerically well-behaved a matrix is. |
| `MapTopK` | Gets or sets the number of top items to consider when calculating Mean Average Precision (MAP). |
| `MaxVIF` | Gets or sets the maximum allowed Variance Inflation Factor (VIF), which measures how much the variance of a regression coefficient is increased due to multicollinearity. |
| `MulticollinearityThreshold` | Gets or sets the correlation threshold above which two variables are considered to have problematic multicollinearity. |
| `NdcgTopK` | Gets or sets the number of top items to consider when calculating Normalized Discounted Cumulative Gain (NDCG). |
| `PacfMaxLag` | Gets or sets the maximum lag to use for the PACF calculation. |

