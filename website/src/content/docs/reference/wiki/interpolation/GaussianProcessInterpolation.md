---
title: "GaussianProcessInterpolation<T>"
description: "Implements Gaussian Process interpolation for one-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Gaussian Process interpolation for one-dimensional data points.

## How It Works

Gaussian Process interpolation is a probabilistic approach to interpolation that not only
provides estimates for unknown points but also quantifies the uncertainty in those estimates.

**For Beginners:** This class helps you estimate values between known data points using a technique
that's especially good when your data might contain noise or uncertainty. Think of it like
drawing a smooth line through your points, but also showing a "confidence band" around that
line to indicate how certain the estimates are. It's particularly useful when you have limited
data or when you want to know how reliable your estimates are.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianProcessInterpolation(Vector<>,Vector<>)` | Creates a new Gaussian Process interpolation from the given data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate()` | Calculates the interpolated y-value for a given x-value using Gaussian Process interpolation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpr` | The Gaussian Process Regression model used for interpolation. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the data points (independent variable). |
| `_y` | The y-coordinates of the data points (dependent variable). |

