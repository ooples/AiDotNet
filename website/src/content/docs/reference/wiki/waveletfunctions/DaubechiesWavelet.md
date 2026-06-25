---
title: "DaubechiesWavelet<T>"
description: "Implements Daubechies wavelets, which are a family of orthogonal wavelets characterized by a maximal number of vanishing moments for a given support width."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements Daubechies wavelets, which are a family of orthogonal wavelets characterized by
a maximal number of vanishing moments for a given support width.

## For Beginners

Daubechies wavelets are like special mathematical magnifying glasses that can analyze
signals at different levels of detail. They're named after Ingrid Daubechies, a mathematician
who made groundbreaking contributions to wavelet theory.

Key features of Daubechies wavelets:

- They have compact support (affect only a limited region)
- They can have a specified number of vanishing moments
- They're orthogonal (no redundancy in the transform)
- They're asymmetric (unlike some other wavelets)

"Vanishing moments" means the wavelet can ignore certain polynomial trends in the data.
For example, a wavelet with 2 vanishing moments will be "blind" to constant and linear trends,
allowing it to focus on more complex patterns.

These wavelets are particularly useful for:

- Image compression (JPEG2000 uses them)
- Signal denoising
- Feature extraction
- Data compression

The order parameter (typically denoted as D2, D4, D6, etc., where the number is twice the order)
controls how many vanishing moments the wavelet has, with higher orders providing more
vanishing moments but wider support.

## How It Works

Daubechies wavelets, named after mathematician Ingrid Daubechies, are a family of orthogonal
wavelets with compact support and a maximal number of vanishing moments for a given support width.
They are widely used in signal processing, image compression, and numerical analysis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DaubechiesWavelet(Int32)` | Initializes a new instance of the DaubechiesWavelet class with the specified order. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the value of the Daubechies wavelet function at point x. |
| `CascadeAlgorithm(Double,Int32)` | Implements the cascade algorithm to approximate the scaling function at point t. |
| `ComputeScalingCoefficients` | Computes the scaling function coefficients for the Daubechies wavelet. |
| `ComputeWaveletCoefficients` | Computes the wavelet function coefficients for the Daubechies wavelet. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the Daubechies wavelet. |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the Daubechies wavelet. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the Daubechies wavelet. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_order` | The order of the Daubechies wavelet. |
| `_scalingCoefficients` | The scaling function coefficients of the Daubechies wavelet. |
| `_waveletCoefficients` | The wavelet function coefficients of the Daubechies wavelet. |

