---
title: "PaulWavelet<T>"
description: "Represents a Paul wavelet function implementation for complex signal analysis and processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Paul wavelet function implementation for complex signal analysis and processing.

## For Beginners

The Paul wavelet is a special mathematical tool for analyzing phases and oscillations in data.

Think of the Paul wavelet like a specialized detector that:

- Can identify not just when oscillations occur, but also their phase (position in the cycle)
- Is particularly good at finding short-lived patterns in your data
- Works with complex numbers (numbers with both real and imaginary parts)

This wavelet is especially useful when you need to track how phases change over time,
such as in brain wave analysis, fluid dynamics, or geophysical signal processing.
Unlike some other wavelets, the Paul wavelet is asymmetric, which makes it particularly
sensitive to the direction of changes in your data.

## How It Works

The Paul wavelet is a complex-valued wavelet that belongs to the family of analytic wavelets.
It is particularly useful for analyzing oscillatory behaviors and phase information in signals.
The Paul wavelet is defined in terms of complex functions and provides good time resolution
with moderate frequency resolution, making it suitable for extracting instantaneous phases
and detecting transient phenomena in signals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PaulWavelet(Int32)` | Initializes a new instance of the `PaulWavelet` class with the specified order. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the Paul wavelet function value at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal using the Paul wavelet transform. |
| `GetScalingCoefficients` | Gets the scaling coefficients (low-pass filter) used in the Paul wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients (high-pass filter) used in the Paul wavelet transform. |
| `PaulFourierTransform()` | Calculates the Fourier transform of the Paul wavelet at the specified frequency. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_fft` | Provides Fast Fourier Transform capabilities for frequency domain analysis. |
| `_order` | The order parameter that controls the properties of the Paul wavelet. |

