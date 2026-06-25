---
title: "Interpolation2DTo1DAdapter<T>"
description: "Adapts a two-dimensional interpolation to work as a one-dimensional interpolation by fixing one coordinate."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Adapts a two-dimensional interpolation to work as a one-dimensional interpolation by fixing one coordinate.

## How It Works

This adapter allows you to use a 2D interpolation method as if it were a 1D interpolation
by keeping one of the coordinates (either X or Y) at a fixed value.

**For Beginners:** Think of this like taking a slice through a 3D surface. Imagine a landscape
with hills and valleys - if you cut through it in a straight line, you get a 2D profile
showing the heights along that line. This adapter does something similar - it takes a 2D
interpolation (which works with X and Y coordinates) and creates a 1D view of it (which
only needs one coordinate) by fixing either the X or Y value.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Interpolation2DTo1DAdapter(I2DInterpolation<>,,Boolean)` | Creates a new adapter that converts a 2D interpolation to a 1D interpolation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets a human-readable description of this interpolation adapter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate()` | Performs one-dimensional interpolation by calling the underlying two-dimensional interpolation with one coordinate fixed. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_fixedCoordinate` | The value of the coordinate that remains constant. |
| `_interpolation2D` | The underlying two-dimensional interpolation method. |
| `_isXFixed` | Indicates whether the X coordinate is fixed (true) or the Y coordinate is fixed (false). |

