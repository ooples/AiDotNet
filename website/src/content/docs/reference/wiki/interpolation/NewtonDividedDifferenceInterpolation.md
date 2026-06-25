---
title: "NewtonDividedDifferenceInterpolation<T>"
description: "Implements Newton's divided difference interpolation method, which creates a polynomial that passes through all given data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Newton's divided difference interpolation method, which creates a polynomial that passes through all given data points.

## For Beginners

Interpolation is a way to estimate values between known data points. Newton's divided difference
interpolation creates a smooth curve (a polynomial) that passes exactly through all your known data points.

## How It Works

Unlike simpler methods like nearest neighbor, this method creates a continuous curve that can provide more
accurate estimates between your data points. It's especially useful when you need a smooth function that
exactly matches your known data.

Think of it like connecting dots with a smooth curve instead of straight lines or steps.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NewtonDividedDifferenceInterpolation(Vector<>,Vector<>)` | Initializes a new instance of the `NewtonDividedDifferenceInterpolation` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCoefficients(Vector<>,Vector<>)` | Calculates the coefficients for Newton's divided difference polynomial. |
| `Interpolate()` | Performs Newton's divided difference interpolation to estimate a y-value for the given x-value. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The coefficients of the Newton polynomial, calculated from the input data. |
| `_numOps` | Operations for performing numeric calculations with generic type T. |
| `_x` | The x-coordinates of the known data points. |

