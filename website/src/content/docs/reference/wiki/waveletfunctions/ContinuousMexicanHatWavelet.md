---
title: "ContinuousMexicanHatWavelet<T>"
description: "Implements the Mexican Hat wavelet (also known as the Ricker wavelet or the second derivative of a Gaussian), which is commonly used for continuous wavelet transforms and feature detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements the Mexican Hat wavelet (also known as the Ricker wavelet or the second derivative of a Gaussian),
which is commonly used for continuous wavelet transforms and feature detection.

## For Beginners

The Mexican Hat wavelet looks like a sombrero or a mountain with valleys on each side.
This shape makes it excellent at finding "bumps" or peaks in your data.

Key features of the Mexican Hat wavelet:

- It has a central positive peak with negative valleys on either side
- It's symmetric (the same on both sides of the center)
- It's good at detecting sudden changes or peaks in signals
- It has no scaling function in the traditional sense

Think of it as a template that you slide over your data, looking for places where
the data has a similar "bump" shape. When the wavelet aligns with a bump in your data,
it produces a strong response.

These wavelets are particularly useful for:

- Finding peaks in spectra
- Edge detection in images
- Scale-space analysis
- Feature detection in various signals
- Analyzing data where you need to identify local maxima or minima

Unlike some other wavelets, the Mexican Hat is primarily used for continuous wavelet
transforms rather than discrete transforms, though this implementation provides
both capabilities.

## How It Works

The Mexican Hat wavelet is the negative normalized second derivative of a Gaussian function.
It has a central peak with symmetric valleys on either side, resembling a Mexican hat.
This wavelet is particularly useful for detecting peaks and edges in signals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContinuousMexicanHatWavelet` | Initializes a new instance of the ContinuousMexicanHatWavelet class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the value of the Mexican Hat wavelet function at point x. |
| `Convolve(Vector<>,Vector<>)` | Performs convolution of an input signal with a filter. |
| `ConvolveReversed(Vector<>,Vector<>)` | Convolves a signal with a time-reversed kernel. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the Mexican Hat wavelet. |
| `Downsample(Vector<>,Int32)` | Downsamples a signal by keeping only every nth sample. |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the Mexican Hat wavelet. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the Mexican Hat wavelet. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `Upsample(Vector<>,Int32)` | Upsamples a signal by inserting zeros between samples. |

