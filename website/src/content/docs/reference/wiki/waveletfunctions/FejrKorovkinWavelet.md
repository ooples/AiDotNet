---
title: "FejérKorovkinWavelet<T>"
description: "Represents a Fejér-Korovkin wavelet function implementation for signal processing and analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Fejér-Korovkin wavelet function implementation for signal processing and analysis.

## For Beginners

A wavelet is a special type of mathematical function that can help analyze data.

Think of wavelets like special magnifying glasses that can zoom in on different parts of your data:

- They can detect patterns at different scales (big patterns and small details)
- They're great for analyzing signals that change over time (like sound or sensor readings)
- They can compress data while preserving important features

The Fejér-Korovkin wavelet is a specific type of wavelet with smooth properties that make it
useful for various applications in signal processing, image analysis, and data compression.

## How It Works

The Fejér-Korovkin wavelet is a mathematical function used in signal processing for decomposing
signals into different frequency components. This implementation supports various orders of the
wavelet and provides methods for calculating wavelet values and decomposing signals using the
wavelet transform.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FejérKorovkinWavelet(Int32)` | Initializes a new instance of the `FejérKorovkinWavelet` class with the specified order. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the wavelet function value at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the wavelet transform. |
| `GetFejérKorovkinCoefficients(Int32)` | Calculates the Fejér-Korovkin coefficients for the specified order. |
| `GetScalingCoefficients` | Gets the scaling coefficients used in the wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients used in the wavelet transform. |
| `InitializeCoefficients` | Initializes the scaling and wavelet coefficients used for signal decomposition. |
| `NormalizeCoefficients(Vector<>)` | Normalizes a set of coefficients to ensure they have unit energy. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `ScalingFunction()` | Evaluates the scaling function at the specified point. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The Fejér-Korovkin wavelet coefficients. |
| `_order` | The order of the Fejér-Korovkin wavelet. |
| `_scalingCoefficients` | The scaling coefficients used for signal decomposition. |
| `_waveletCoefficients` | The wavelet coefficients used for signal decomposition. |

