---
title: "BiorthogonalWavelet<T>"
description: "Implements biorthogonal wavelets, which offer symmetry and linear phase properties while maintaining perfect reconstruction capabilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements biorthogonal wavelets, which offer symmetry and linear phase properties while maintaining
perfect reconstruction capabilities.

## For Beginners

Biorthogonal wavelets are special mathematical tools that offer more flexibility than standard wavelets.

The key difference is that biorthogonal wavelets use:

- One set of functions to break down (decompose) a signal
- A different but related set of functions to rebuild (reconstruct) it

This approach offers several advantages:

- Perfect reconstruction: The signal can be rebuilt exactly without errors
- Symmetry: The wavelets can be symmetric, which reduces edge artifacts
- Linear phase: Important for preserving the shape of features in the signal

These properties make biorthogonal wavelets particularly useful for:

- Image compression (JPEG2000 uses them)
- Signal denoising where preserving edges is important
- Applications where phase information matters

You can think of biorthogonal wavelets as using two complementary lenses - one for analyzing
and one for synthesizing - that work together perfectly.

## How It Works

Biorthogonal wavelets use different basis functions for decomposition and reconstruction, allowing them
to achieve properties that are impossible with orthogonal wavelets. They are particularly useful in
applications where symmetry and linear phase are important, such as image processing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiorthogonalWavelet(Int32,Int32)` | Initializes a new instance of the BiorthogonalWavelet class with the specified decomposition and reconstruction orders. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the value of the biorthogonal wavelet function at point x. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the biorthogonal wavelet. |
| `GetDecompositionCoefficients(Int32)` | Gets the decomposition coefficients for the specified order. |
| `GetDecompositionHighPassFilter` | Gets the high-pass filter coefficients used for decomposition. |
| `GetDecompositionLowPassFilter` | Gets the low-pass filter coefficients used for decomposition. |
| `GetReconstructionCoefficients(Int32)` | Gets the reconstruction coefficients for the specified order. |
| `GetReconstructionHighPassFilter` | Gets the high-pass filter coefficients used for reconstruction. |
| `GetReconstructionLowPassFilter` | Gets the low-pass filter coefficients used for reconstruction. |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the biorthogonal wavelet. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the biorthogonal wavelet. |
| `NormalizeAndConvert(Double[])` | Normalizes and converts an array of double coefficients to type T. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `ScalingFunction()` | Evaluates the basic scaling function at point x. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decompositionCoefficients` | Coefficients used for the decomposition process. |
| `_decompositionOrder` | The order of the wavelet used for decomposition. |
| `_reconstructionCoefficients` | Coefficients used for the reconstruction process. |
| `_reconstructionOrder` | The order of the wavelet used for reconstruction. |

