---
title: "CubicSplineInterpolation<T>"
description: "Implements cubic spline interpolation for one-dimensional data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements cubic spline interpolation for one-dimensional data points.

## How It Works

Cubic spline interpolation creates a smooth curve that passes through all given data points.
The curve consists of piecewise cubic polynomials with continuous first and second derivatives.

**For Beginners:** This class helps you estimate values between known data points.
Imagine you have measurements at specific times (like temperature readings every hour),
and you want to estimate what happened between those measurements. Cubic spline
interpolation creates a smooth curve that passes through all your known points and
provides natural-looking estimates for the points in between. It's like connecting
dots with a flexible curve rather than straight lines.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CubicSplineInterpolation(Vector<>,Vector<>)` | Creates a new cubic spline interpolation from the given data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCoefficients` | Calculates the coefficients of the cubic spline polynomials. |
| `FindInterval()` | Finds the index of the interval in the x-array that contains the given x-value. |
| `Interpolate()` | Calculates the interpolated y-value for a given x-value using cubic spline interpolation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_a` | The constant coefficients of the cubic polynomials (equal to the y values). |
| `_b` | The coefficients of the linear terms in the cubic polynomials. |
| `_c` | The coefficients of the quadratic terms in the cubic polynomials. |
| `_d` | The coefficients of the cubic terms in the cubic polynomials. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the data points (independent variable). |
| `_y` | The y-coordinates of the data points (dependent variable). |

