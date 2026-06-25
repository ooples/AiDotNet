---
title: "ITimeSeriesDecomposition<T>"
description: "Defines methods and properties for decomposing time series data into its component parts."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines methods and properties for decomposing time series data into its component parts.

## How It Works

Time series decomposition breaks down a sequence of data points into underlying patterns
such as trend, seasonality, and residual components.

**For Beginners:** Time series decomposition is like taking apart a complex signal (like sales data
over time) into simpler pieces that are easier to understand. Imagine your store's sales
throughout the year - decomposition helps you separate:

- The overall growth trend (are sales generally increasing?)
- Seasonal patterns (higher sales during holidays?)
- Day-to-day random variations

This makes it easier to understand what's really happening in your data and make better predictions.

## Properties

| Property | Summary |
|:-----|:--------|
| `TimeSeries` | Gets the original time series data that was decomposed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetComponent(DecompositionComponentType)` | Gets a specific component of the time series decomposition. |
| `GetComponents` | Gets all available decomposition components as a dictionary. |
| `HasComponent(DecompositionComponentType)` | Checks if a specific decomposition component is available. |

