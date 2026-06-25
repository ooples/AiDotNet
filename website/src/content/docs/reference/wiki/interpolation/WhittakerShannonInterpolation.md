---
title: "WhittakerShannonInterpolation<T>"
description: "Implements the Whittaker-Shannon interpolation method, also known as sinc interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpolation`

Implements the Whittaker-Shannon interpolation method, also known as sinc interpolation.

## For Beginners

Think of this interpolation like recreating a smooth curve from a series of dots.
Imagine you have a photograph that's been converted to a grid of pixels. This method helps you
"zoom in" between those pixels to create a smoother, higher-resolution image. It works best when
your data points are evenly spaced (like pixels in a digital image) and when the underlying pattern
doesn't contain frequencies that are too high (meaning the data doesn't wiggle up and down too rapidly).

## How It Works

Whittaker-Shannon interpolation is a technique based on the sampling theorem, which states that
a band-limited function can be perfectly reconstructed from its samples if the sampling rate
is at least twice the highest frequency in the function.

This method is particularly useful for signal processing applications, such as audio or image processing,
where you need to reconstruct continuous signals from discrete samples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WhittakerShannonInterpolation(Vector<>,Vector<>)` | Initializes a new instance of Whittaker-Shannon interpolation with the specified data points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate()` | Interpolates a y-value for the given x coordinate using Whittaker-Shannon interpolation. |
| `IsUniformlySampled` | Checks if the input data points are uniformly sampled (evenly spaced). |
| `Sinc()` | Calculates the sinc function value for a given input. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_x` | The x-coordinates of the data points. |
| `_y` | The y-coordinates (values) of the data points. |

