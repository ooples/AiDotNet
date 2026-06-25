---
title: "BattleLemarieWavelet<T>"
description: "Implements the Battle-Lemarie wavelet function, which is a smooth, orthogonal wavelet based on B-splines."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements the Battle-Lemarie wavelet function, which is a smooth, orthogonal wavelet based on B-splines.

## For Beginners

A wavelet is a mathematical function used to divide a signal into different frequency components.
Think of it as a special kind of lens that lets you examine different details of a signal.

The Battle-Lemarie wavelet is particularly useful because:

- It's smooth, which helps avoid artifacts in signal processing
- It has good localization in both time and frequency domains
- It can be adjusted (via the order parameter) to balance between time and frequency resolution

You might use this wavelet for:

- Image compression
- Noise reduction
- Feature extraction from signals
- Analyzing signals with different scales of detail

The higher the order, the smoother the wavelet, but also the wider its support (meaning it
considers more neighboring points when analyzing a signal).

## How It Works

The Battle-Lemarie wavelet is a family of orthogonal wavelets constructed from B-spline functions.
It offers good frequency localization and smoothness properties, making it useful for signal analysis
where both time and frequency resolution are important.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BattleLemarieWavelet(Int32)` | Initializes a new instance of the BattleLemarieWavelet class with the specified order. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BSplineFourier(,Int32)` | Calculates the Fourier transform of the B-spline function at a given frequency. |
| `Calculate()` | Calculates the value of the Battle-Lemarie wavelet function at point x. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the Battle-Lemarie wavelet. |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the Battle-Lemarie wavelet. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the Battle-Lemarie wavelet. |
| `InverseFFT(Vector<Complex<>>)` | Performs an inverse Fast Fourier Transform (FFT) on the input complex vector. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `SumSquaredBSplineFourier(,Int32)` | Calculates the sum of squared Fourier transforms of shifted B-splines. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_order` | The order of the B-spline used to construct the wavelet. |

