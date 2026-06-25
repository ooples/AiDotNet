---
title: "NearestNeighborInterpolation<T>"
description: "Implements nearest neighbor interpolation, a simple method that finds the closest known data point and returns its corresponding value."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements nearest neighbor interpolation, a simple method that finds the closest known data point
and returns its corresponding value.

## For Beginners

Interpolation is a way to estimate values between known data points. Imagine you have
a set of (x,y) points on a graph, and you want to find the y-value for an x that isn't in your original data.
Nearest neighbor interpolation simply finds the closest x-value in your data and returns its corresponding y-value.

## How It Works

This is the simplest form of interpolation and works like a "staircase" function rather than a smooth curve.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NearestNeighborInterpolation(Vector<>,Vector<>)` | Initializes a new instance of the `NearestNeighborInterpolation` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FindNearestIndex()` | Finds the index of the data point whose x-value is closest to the given value. |
| `Interpolate()` | Performs nearest neighbor interpolation to estimate a y-value for the given x-value. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with generic type T. |
| `_x` | The x-coordinates of the known data points. |
| `_y` | The y-coordinates (values) of the known data points. |

