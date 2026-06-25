---
title: "MexicanHatWavelet<T>"
description: "Represents a Mexican Hat wavelet function implementation for signal processing and analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Mexican Hat wavelet function implementation for signal processing and analysis.

## For Beginners

The Mexican Hat wavelet looks like a sombrero or a bell curve with a dip in the middle.

Think of the Mexican Hat wavelet like a special pattern-matching tool that:

- Has a distinctive shape with a central peak surrounded by two valleys, then tapering to zero
- Is excellent at detecting edges, boundaries, and "blob-like" features in your data
- Gets its name because its shape resembles a Mexican sombrero hat viewed from above

This wavelet is particularly useful for finding places where your data changes its direction
twice in a row (like going up, then down, then up again). This makes it ideal for detecting
objects in images, finding peaks in signals, or identifying boundaries between different regions.

## How It Works

The Mexican Hat wavelet, also known as the Ricker wavelet or the second derivative of the Gaussian function,
is a wavelet consisting of a negative normalized second derivative of a Gaussian function.
It is commonly used in image processing, computer vision, and various signal analysis applications
due to its ability to detect transitions and edges at multiple scales.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MexicanHatWavelet(Double)` | Initializes a new instance of the `MexicanHatWavelet` class with the specified sigma. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the Mexican Hat wavelet function value at the specified point. |
| `CalculateDerivative()` | Calculates the derivative of the Mexican Hat function at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal using the Mexican Hat wavelet and its derivative. |
| `GetScalingCoefficients` | Gets the scaling coefficients (Mexican Hat function) used in the wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients (derivative of Mexican Hat function) used in the wavelet transform. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_sigma` | The standard deviation parameter that controls the width of the Mexican Hat wavelet. |

