---
title: "CoifletWavelet<T>"
description: "Implements Coiflet wavelets, which are compactly supported wavelets with a high number of vanishing moments for both the wavelet and scaling functions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements Coiflet wavelets, which are compactly supported wavelets with a high number of vanishing moments
for both the wavelet and scaling functions.

## For Beginners

Coiflet wavelets are special mathematical tools designed to analyze signals with particular properties.

Key features of Coiflet wavelets:

- They're nearly symmetric (more symmetric than Daubechies wavelets)
- They have vanishing moments for both the wavelet and scaling functions
- They have compact support (affect only a limited region)

"Vanishing moments" means the wavelet can ignore certain polynomial trends in the data.
For example, a wavelet with 3 vanishing moments will be "blind" to constant, linear, and
quadratic trends, allowing it to focus on more complex patterns.

These properties make Coiflet wavelets particularly useful for:

- Signal compression
- Feature extraction
- Numerical analysis
- Applications where symmetry is important

The order parameter (1-5) controls how many vanishing moments the wavelet has,
with higher orders providing more vanishing moments but wider support.

## How It Works

Coiflet wavelets were designed by Ingrid Daubechies at the request of Ronald Coifman. They are distinguished
by having vanishing moments for both the wavelet and scaling functions, which makes them more symmetric
than Daubechies wavelets and better suited for certain applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CoifletWavelet(Int32)` | Initializes a new instance of the CoifletWavelet class with the specified order. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the value of the Coiflet wavelet function at point x. |
| `Convolve(Vector<>,Vector<>)` | Performs convolution of an input signal with a filter. |
| `ConvolveReversed(Vector<>,Vector<>)` | Convolves with a time-reversed filter. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the Coiflet wavelet. |
| `Downsample(Vector<>,Int32)` | Downsamples a signal by keeping only every nth sample. |
| `GetCoifletCoefficients(Int32)` | Gets the pre-calculated coefficients for the Coiflet wavelet of the specified order. |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the Coiflet wavelet. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the Coiflet wavelet. |
| `NormalizeAndConvert(Double[])` | Normalizes and converts an array of double coefficients to type T. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `ScalingFunction(Double)` | Evaluates the scaling function at point t using a recursive approximation. |
| `ScalingFunctionRecursive(Double,Int32)` | Recursive helper for scaling function with depth limit to prevent stack overflow. |
| `Upsample(Vector<>,Int32)` | Upsamples a signal by inserting zeros. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The coefficients of the Coiflet wavelet. |
| `_order` | The order of the Coiflet wavelet. |

