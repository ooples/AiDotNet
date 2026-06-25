---
title: "DOGWavelet<T>"
description: "Implements the Derivative of Gaussian (DOG) wavelet, which is based on the nth derivative of the Gaussian function and is useful for detecting changes in signals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements the Derivative of Gaussian (DOG) wavelet, which is based on the nth derivative
of the Gaussian function and is useful for detecting changes in signals.

## For Beginners

The Derivative of Gaussian (DOG) wavelet is like a mathematical tool that's especially
good at finding places where your data changes quickly.

Key features of DOG wavelets:

- They're based on derivatives of the Gaussian function (bell curve)
- Different orders detect different types of changes
- They're symmetric (the same on both sides of the center)
- They have good localization in both time and frequency

Think of them as detectors that respond strongly when your data shows specific
patterns of change:

- 1st order (order=1): Detects edges (sudden jumps)
- 2nd order (order=2): Detects peaks and valleys (Mexican Hat wavelet)
- Higher orders: Detect more complex patterns of change

These wavelets are particularly useful for:

- Edge detection in signals and images
- Finding points where data changes rapidly
- Scale-space analysis
- Feature detection
- Signal analysis where transitions are important

The order parameter lets you choose which type of change you're looking for.

## How It Works

The Derivative of Gaussian (DOG) wavelet is derived from taking derivatives of the Gaussian function.
It has excellent localization properties in both time and frequency domains and is particularly
useful for detecting changes or transitions in signals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DOGWavelet(Int32)` | Initializes a new instance of the DOGWavelet class with the specified order. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the value of the DOG wavelet function at point x. |
| `Convolve(Vector<>,Vector<>)` | Performs convolution of an input signal with a filter. |
| `ConvolveReversed(Vector<>,Vector<>)` | Convolves with a time-reversed filter. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the DOG wavelet. |
| `Downsample(Vector<>,Int32)` |  |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the DOG wavelet. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the DOG wavelet. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `Upsample(Vector<>,Int32)` | Upsamples a signal by inserting zeros. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_order` | The order of the derivative of the Gaussian function. |

