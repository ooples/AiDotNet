---
title: "MeyerWavelet<T>"
description: "Represents a Meyer wavelet function implementation for frequency domain analysis and signal processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Meyer wavelet function implementation for frequency domain analysis and signal processing.

## For Beginners

The Meyer wavelet is a special mathematical tool for analyzing signals in the frequency domain.

Think of the Meyer wavelet like a musical tuning fork that:

- Works especially well for analyzing what frequencies are present in your data
- Has very clean frequency separation (it doesn't mix up different frequencies)
- Is infinitely smooth, which makes it good for analyzing continuous signals

Unlike other wavelets that work directly with your data points (time domain), the Meyer wavelet
works with the frequencies in your data (frequency domain). This is like looking at a piece of music
as a collection of different notes played simultaneously, rather than as sounds that change over time.

## How It Works

The Meyer wavelet is a frequency domain wavelet that is infinitely differentiable with compact
support in the frequency domain. Unlike many other wavelets that are defined primarily in the
time domain, the Meyer wavelet is more naturally defined in the frequency domain, making it
particularly useful for spectral analysis. This implementation uses Fast Fourier Transform (FFT)
for efficient computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeyerWavelet` | Initializes a new instance of the `MeyerWavelet` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AuxiliaryFunction(Double)` | Provides a smooth transition function for the Meyer wavelet. |
| `Calculate()` | Calculates the Meyer wavelet function value at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal using the Meyer wavelet transform. |
| `GetScalingCoefficients` | Gets the scaling coefficients (low-pass filter) used in the Meyer wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients (high-pass filter) used in the Meyer wavelet transform. |
| `MeyerFunction(Double)` | Implements the Meyer wavelet function for time domain calculation. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `Vf(Double)` | Provides a polynomial function used in the frequency domain definitions. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_fft` | Provides Fast Fourier Transform capabilities for frequency domain analysis. |

