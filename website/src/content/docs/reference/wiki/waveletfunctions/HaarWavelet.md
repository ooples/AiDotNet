---
title: "HaarWavelet<T>"
description: "Represents a Haar wavelet function implementation for signal processing and analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Haar wavelet function implementation for signal processing and analysis.

## For Beginners

The Haar wavelet is like a digital "on/off" switch for analyzing data.

Think of the Haar wavelet as the simplest possible pattern-matching tool:

- It's a square wave that is +1 for half its width and -1 for the other half
- It's excellent at detecting sudden changes or edges in your data
- It's the oldest and simplest type of wavelet, discovered in 1909

The Haar wavelet is particularly good at finding abrupt transitions in your data,
like the edge between a light and dark area in an image or a sudden change in a sound wave.
It's widely used in image compression, feature detection, and as a teaching tool for wavelet concepts.

## How It Works

The Haar wavelet is the simplest possible wavelet, resembling a step function. It is
discontinuous and represents the same wavelet as the Daubechies wavelet with one vanishing moment.
This implementation provides methods for calculating wavelet values and decomposing signals
using the Haar wavelet transform.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HaarWavelet` | Initializes a new instance of the `HaarWavelet` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the Haar wavelet function value at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal using the Haar wavelet transform. |
| `GetScalingCoefficients` | Gets the scaling coefficients used in the Haar wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients used in the Haar wavelet transform. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |

