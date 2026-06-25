---
title: "WaveletDecomposition<T>"
description: "Implements wavelet-based decomposition methods for time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Implements wavelet-based decomposition methods for time series data.

## For Beginners

Wavelet decomposition is like breaking down a complex signal (like music)
into different frequency bands. Think of it as separating bass, mid-range, and treble in music.
This helps identify patterns at different time scales - from long-term trends to short-term fluctuations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WaveletDecomposition(Vector<>,IWaveletFunction<>,Int32,WaveletAlgorithmType,Int32)` | Initializes a new instance of the WaveletDecomposition class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decompose` | Performs the wavelet decomposition using the selected algorithm. |
| `DecomposeDWT` | Performs Discrete Wavelet Transform (DWT) decomposition. |
| `DecomposeMODWT` | Performs Maximal Overlap Discrete Wavelet Transform (MODWT) decomposition. |
| `DecomposeSWT` | Performs Stationary Wavelet Transform (SWT) decomposition. |
| `MODWTStep(Vector<>,Int32)` | Performs a single step of the MODWT decomposition. |
| `PadToLength(Vector<>,Int32)` | Extends a vector to the specified length by padding with zeros. |
| `SWTStep(Vector<>,Int32)` | Performs a single step of the Stationary Wavelet Transform (SWT) decomposition. |
| `UpsampleFilter(Vector<>,Int32)` | Expands a filter by inserting zeros between its elements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_algorithm` | The type of wavelet algorithm to use. |
| `_levels` | The number of decomposition levels to perform. |
| `_minimumDecompositionLength` | The minimum length of data required to continue decomposition. |
| `_wavelet` | The wavelet function used for decomposition. |

