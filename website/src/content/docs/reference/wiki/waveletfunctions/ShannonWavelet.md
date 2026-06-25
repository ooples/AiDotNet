---
title: "ShannonWavelet<T>"
description: "Represents a Shannon wavelet function implementation for signal processing and frequency analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Shannon wavelet function implementation for signal processing and frequency analysis.

## For Beginners

The Shannon wavelet is like a specialized frequency analyzer.

Think of the Shannon wavelet as a musical tuning fork that:

- Can precisely identify specific frequencies in your data
- Has perfect frequency localization (it knows exactly which frequencies are present)
- Is less precise about when those frequencies occur in the signal

This wavelet is particularly useful when you need to know exactly which frequencies are
in your data, but don't need to know precisely when they occur. It's like having a
perfect pitch detector that can tell you exactly which notes are being played, but is
less precise about when each note starts and stops.

## How It Works

The Shannon wavelet is a band-limited wavelet that is perfectly localized in the frequency domain.
It is defined as the product of a sinc function and a cosine modulation. This wavelet is particularly
useful for signal analysis where precise frequency localization is required, though it has poor
time localization due to the slow decay of the sinc function.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShannonWavelet` | Initializes a new instance of the `ShannonWavelet` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the Shannon wavelet function value at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal using the Shannon wavelet transform. |
| `GetScalingCoefficients` | Gets the scaling coefficients used in the Shannon wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients used in the Shannon wavelet transform. |
| `Reconstruct(Vector<>,Vector<>,Nullable<Int32>)` | Reconstructs the original signal from approximation and detail coefficients. |

