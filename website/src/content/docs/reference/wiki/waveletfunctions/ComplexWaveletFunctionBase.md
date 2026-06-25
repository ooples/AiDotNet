---
title: "ComplexWaveletFunctionBase<T>"
description: "Base class for all complex wavelet function implementations providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.WaveletFunctions`

Base class for all complex wavelet function implementations providing common functionality.

## For Beginners

This is a foundation class that all complex wavelets build upon.

Think of this base class like a blueprint that ensures all complex wavelets have:

- Access to mathematical operations for both real and complex numbers
- A consistent structure that makes them work together
- Shared utilities that every complex wavelet needs

Complex wavelets work with complex numbers (numbers with real and imaginary parts),
which allows them to capture both amplitude and phase information in signals.
This is especially useful for analyzing oscillatory patterns.

## How It Works

This abstract base class provides shared infrastructure for complex wavelet function implementations,
including numeric operations support for both the underlying type T and complex operations.
All complex wavelet functions in the library should inherit from this base class to ensure
consistent behavior and reduce code duplication.

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Complex<>)` | Calculates the complex wavelet function value at the specified complex point. |
| `Convolve(Vector<Complex<>>,Vector<Complex<>>)` | Performs convolution of a complex input signal with a complex filter. |
| `Decompose(Vector<Complex<>>)` | Decomposes a complex input signal using the wavelet transform. |
| `Downsample(Vector<Complex<>>,Int32)` | Downsamples a complex signal by keeping only every nth sample. |
| `GetScalingCoefficients` | Gets the complex scaling coefficients used in the wavelet transform. |
| `GetWaveletCoefficients` | Gets the complex wavelet coefficients used in the wavelet transform. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the underlying numeric type T. |

