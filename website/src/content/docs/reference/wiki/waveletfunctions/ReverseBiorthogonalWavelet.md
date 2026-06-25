---
title: "ReverseBiorthogonalWavelet<T>"
description: "Represents a Reverse Biorthogonal wavelet function implementation for signal processing and analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Reverse Biorthogonal wavelet function implementation for signal processing and analysis.

## For Beginners

Reverse Biorthogonal wavelets are specialized mathematical tools for analyzing data.

Think of Reverse Biorthogonal wavelets like precise measuring instruments that:

- Can analyze your data while preserving its exact shape and features
- Work especially well with images and signals where shape matters
- Come in different "sizes" (orders) for different levels of detail

These wavelets are particularly good at preserving the symmetry and shape of features in your data.
This makes them excellent for applications like image compression, where you want to reduce file
size while maintaining visual quality, or in medical imaging where preserving exact shapes is crucial.

## How It Works

The Reverse Biorthogonal wavelet is a family of symmetric wavelets that provide exact reconstruction
while having symmetric decomposition and reconstruction filters. These wavelets are particularly
useful in image processing and applications where phase information is important. This implementation
supports various orders of the wavelet and different boundary handling methods.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReverseBiorthogonalWavelet(WaveletType,BoundaryHandlingMethod,Int32)` | Initializes a new instance of the `ReverseBiorthogonalWavelet` class with the specified parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the wavelet function value at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the wavelet transform. |
| `DecomposeChunk(Vector<>,Vector<>,Vector<>,Int32,Int32)` | Processes a chunk of the input signal during decomposition. |
| `DecomposeMultiLevel(Vector<>,Int32)` | Performs multi-level decomposition of a signal using the wavelet transform. |
| `DiscreteCascadeAlgorithm()` | Implements the discrete cascade algorithm to approximate the continuous scaling function. |
| `GetExtendedIndex(Int32,Int32)` | Extends an index beyond the boundaries of an array according to the selected boundary handling method. |
| `GetReverseBior11Coefficients` | Gets the filter coefficients for the ReverseBior11 wavelet. |
| `GetReverseBior13Coefficients` | Gets the filter coefficients for the ReverseBior13 wavelet. |
| `GetReverseBior22Coefficients` | Gets the filter coefficients for the ReverseBior22 wavelet. |
| `GetReverseBior24Coefficients` | Gets the filter coefficients for the ReverseBior24 wavelet. |
| `GetReverseBior26Coefficients` | Gets the filter coefficients for the ReverseBior26 wavelet. |
| `GetReverseBior28Coefficients` | Gets the filter coefficients for the ReverseBior28 wavelet. |
| `GetReverseBior31Coefficients` | Gets the filter coefficients for the ReverseBior31 wavelet. |
| `GetReverseBior33Coefficients` | Gets the filter coefficients for the ReverseBior33 wavelet. |
| `GetReverseBior35Coefficients` | Gets the filter coefficients for the ReverseBior35 wavelet. |
| `GetReverseBior37Coefficients` | Gets the filter coefficients for the ReverseBior37 wavelet. |
| `GetReverseBior39Coefficients` | Gets the filter coefficients for the ReverseBior39 wavelet. |
| `GetReverseBior44Coefficients` | Gets the filter coefficients for the ReverseBior44 wavelet. |
| `GetReverseBior46Coefficients` | Gets the filter coefficients for the ReverseBior46 wavelet. |
| `GetReverseBior48Coefficients` | Gets the filter coefficients for the ReverseBior48 wavelet. |
| `GetReverseBior55Coefficients` | Gets the filter coefficients for the ReverseBior55 wavelet. |
| `GetReverseBior68Coefficients` | Gets the filter coefficients for the ReverseBior68 wavelet. |
| `GetReverseBiorthogonalCoefficients(WaveletType)` | Gets the filter coefficients for the specified Reverse Biorthogonal wavelet type. |
| `GetScalingCoefficients` | Gets the scaling coefficients used in the Reverse Biorthogonal wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients used in the Reverse Biorthogonal wavelet transform. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs a signal from its approximation and detail coefficients. |
| `ReconstructChunk(Vector<>,Vector<>,Vector<>,Int32,Int32)` | Processes a chunk of the signal during reconstruction. |
| `ReconstructMultiLevel(Vector<>,List<Vector<>>)` | Performs multi-level reconstruction of a signal from its wavelet transform coefficients. |
| `ScalingFunction()` | Provides a simple initial approximation of the scaling function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_boundaryMethod` | The method used to handle boundary conditions when processing signals of finite length. |
| `_chunkSize` | The size of data chunks used when processing large signals. |
| `_decompositionHighPass` | The high-pass filter coefficients used during signal decomposition. |
| `_decompositionLowPass` | The low-pass filter coefficients used during signal decomposition. |
| `_reconstructionHighPass` | The high-pass filter coefficients used during signal reconstruction. |
| `_reconstructionLowPass` | The low-pass filter coefficients used during signal reconstruction. |
| `_waveletType` | The specific type of Reverse Biorthogonal wavelet being used. |

