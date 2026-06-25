---
title: "RegressionConformalInterval<TOutput>"
description: "Represents a conformal prediction interval for regression-style outputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents a conformal prediction interval for regression-style outputs.

## For Beginners

Instead of a single number, you get a range where the true value is likely to fall.

## How It Works

The interval is typically computed as `[prediction - q, prediction + q]` where `q` is a quantile of calibration residuals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RegressionConformalInterval(,)` | Initializes a new instance of the `RegressionConformalInterval` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Lower` | Gets the lower bound of the interval. |
| `Upper` | Gets the upper bound of the interval. |

