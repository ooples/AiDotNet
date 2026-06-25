---
title: "InterventionAnalysisOptions<T, TInput, TOutput>"
description: "Configuration options for Intervention Analysis, which is a time series modeling technique used to assess the impact of specific events or interventions on a time series."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Intervention Analysis, which is a time series modeling technique used to
assess the impact of specific events or interventions on a time series.

## For Beginners

Intervention Analysis helps you measure how specific events affected your
time series data. For example, if you're tracking daily sales and launched a major marketing campaign
on a specific date, intervention analysis can help you determine how much that campaign actually
boosted your sales while accounting for other factors like seasonal patterns or existing trends.

Think of it like trying to measure the effect of a new medication on a patient's health while
accounting for their normal day-to-day fluctuations in health metrics. Just looking at the raw
numbers before and after might be misleading - intervention analysis gives you a more accurate
picture of the true impact.

This class inherits from TimeSeriesRegressionOptions, so all the general time series regression
settings are also available. The additional settings specific to intervention analysis let you
configure how the model handles both the time series patterns and the intervention effects.

## How It Works

Intervention Analysis extends time series regression by explicitly modeling the effects of known
interventions or events (like policy changes, marketing campaigns, or natural disasters) on a time
series. It combines ARIMA (AutoRegressive Integrated Moving Average) modeling with additional
components that represent these interventions, allowing you to quantify their impact while accounting
for the underlying time series patterns.

## Properties

| Property | Summary |
|:-----|:--------|
| `AROrder` | Gets or sets the order of the AutoRegressive (AR) component in the ARIMA model. |
| `Interventions` | Gets or sets the list of interventions to be analyzed in the time series. |
| `MAOrder` | Gets or sets the order of the Moving Average (MA) component in the ARIMA model. |
| `Optimizer` | Gets or sets the optimizer used to find the best parameters for the intervention analysis model. |

