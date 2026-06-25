---
title: "BSplineWavelet<T>"
description: "Implements a B-spline wavelet, which is a smooth wavelet constructed from B-spline functions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements a B-spline wavelet, which is a smooth wavelet constructed from B-spline functions.

## For Beginners

B-spline wavelets are like smooth, bell-shaped curves that can be used to analyze signals at
different levels of detail.

Key features of B-spline wavelets:

- They're very smooth (no sharp corners or discontinuities)
- They have compact support (affect only a limited region)
- Their smoothness increases with the order

Think of B-splines as building blocks that can be combined to create smooth curves.
Higher-order B-splines are smoother but have wider support (affect more neighboring points).

These wavelets are particularly useful for:

- Signal smoothing and denoising
- Feature extraction where smoothness is important
- Applications where derivatives of the signal are analyzed
- Image processing tasks requiring high regularity

The order parameter lets you control the trade-off between smoothness and localization.

## How It Works

B-spline wavelets are constructed from B-spline functions, which are piecewise polynomial functions
with compact support and maximum smoothness for a given support width. These wavelets offer excellent
smoothness properties and are particularly useful for applications requiring high regularity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BSplineWavelet(Int32)` | Initializes a new instance of the BSplineWavelet class with the specified order. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BSpline(,Int32)` | Calculates the value of the B-spline function of order n at point x. |
| `Calculate()` | Calculates the value of the B-spline wavelet function at point x. |
| `Convolve(Vector<>,Vector<>)` | Performs convolution of an input signal with a filter. |
| `ConvolveReversed(Vector<>,Vector<>)` | Convolves with a time-reversed filter. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the B-spline wavelet. |
| `Downsample(Vector<>,Int32)` | Downsamples a signal by keeping only every nth sample. |
| `GetBSplineCoefficients(Int32)` | Generates B-spline coefficients for the specified order. |
| `GetDecompositionHighPassFilter` | Gets the high-pass filter coefficients used for decomposition. |
| `GetDecompositionLowPassFilter` | Gets the low-pass filter coefficients used for decomposition. |
| `GetReconstructionHighPassFilter` | Gets the high-pass filter coefficients used for reconstruction (synthesis). |
| `GetReconstructionLowPassFilter` | Gets the low-pass filter coefficients used for reconstruction (synthesis). |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the B-spline wavelet. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the B-spline wavelet. |
| `NormalizeAndConvert(Double[])` | Normalizes and converts an array of double coefficients to type T. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `Upsample(Vector<>,Int32)` | Upsamples a signal by inserting zeros. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_order` | The order of the B-spline used to construct the wavelet. |

