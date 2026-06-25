---
title: "GaussianWavelet<T>"
description: "Represents a Gaussian wavelet function implementation for signal processing and analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Gaussian wavelet function implementation for signal processing and analysis.

## For Beginners

A Gaussian wavelet is a smooth bell-shaped curve that helps analyze data.

Think of the Gaussian wavelet like a special magnifying glass that:

- Can detect smooth transitions and gradual changes in your data
- Has a perfect balance between time and frequency precision
- Resembles the familiar "bell curve" shape used in statistics

This type of wavelet is especially good at finding gradual changes and smooth features
in signals like audio recordings, temperature measurements, or image brightness variations.
It's named after Carl Friedrich Gauss, a mathematician who first described the bell-shaped
curve that forms the foundation of this wavelet.

## How It Works

The Gaussian wavelet is based on the Gaussian function and its derivatives. It provides
excellent localization in both time and frequency domains, making it useful for detecting
smooth changes and transitions in signals. This implementation uses the Gaussian function
for approximation coefficients and its first derivative for detail coefficients.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianWavelet(Double)` | Initializes a new instance of the `GaussianWavelet` class with the specified sigma. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the Gaussian function value at the specified point. |
| `CalculateDerivative()` | Calculates the first derivative of the Gaussian function at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal using the Gaussian wavelet and its derivative. |
| `GetScalingCoefficients` | Gets the scaling coefficients (Gaussian function) used in the wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients (derivative of Gaussian function) used in the wavelet transform. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_sigma` | The standard deviation parameter that controls the width of the Gaussian wavelet. |

