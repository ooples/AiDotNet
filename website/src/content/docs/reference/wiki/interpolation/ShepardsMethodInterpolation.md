---
title: "ShepardsMethodInterpolation<T>"
description: "Implements Shepard's Method for interpolating scattered data points in 2D space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements Shepard's Method for interpolating scattered data points in 2D space.

## For Beginners

Imagine you have several points with known heights (like hills on a landscape).
Shepard's Method helps you estimate the height at any other location by considering all known points,
but giving more importance to the closest ones. It's like saying "this unknown point is probably
more similar to nearby points than to faraway points." The power parameter controls how quickly
the influence of distant points diminishes.

## How It Works

Shepard's Method is a form of inverse distance weighting interpolation that creates
a smooth surface passing through all the provided data points.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShepardsMethodInterpolation(Vector<>,Vector<>,Vector<>,Double)` | Initializes a new instance of Shepard's Method interpolation with the specified data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDistance(,,,)` | Calculates the Euclidean distance between two points in 2D space. |
| `Interpolate(,)` | Interpolates a z-value for the given x and y coordinates using Shepard's Method. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_power` | The power parameter that controls how quickly the influence of points decreases with distance. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates of the data points. |
| `_z` | The z-values (heights) at each data point. |

