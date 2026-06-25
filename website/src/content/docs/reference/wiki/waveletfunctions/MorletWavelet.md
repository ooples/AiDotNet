---
title: "MorletWavelet<T>"
description: "Represents a Morlet wavelet function implementation for time-frequency analysis and signal processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Morlet wavelet function implementation for time-frequency analysis and signal processing.

## For Beginners

The Morlet wavelet is like a tunable detector for patterns in your data.

Think of the Morlet wavelet like a musical note with an adjustable pitch that:

- Has a smooth bell-shaped envelope (like a Gaussian curve)
- Contains oscillations (like a cosine wave) inside this envelope
- Can be tuned to detect specific frequencies in your data

This wavelet is particularly good at analyzing signals where you need to know
both when something happens (time localization) and what frequencies are present
(frequency localization). It's commonly used for analyzing audio, brain waves (EEG),
vibrations, and many other types of signals.

## How It Works

The Morlet wavelet is a complex wavelet defined as a plane wave modulated by a Gaussian window.
It offers excellent time-frequency localization and is widely used in signal processing, geophysics,
audio analysis, and various other fields. This implementation supports customization of the central
frequency parameter, which affects the balance between time and frequency resolution.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MorletWavelet(Double)` | Initializes a new instance of the `MorletWavelet` class with the specified omega parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the Morlet wavelet function value at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal using the Morlet wavelet transform. |
| `GetScalingCoefficients` | Gets the scaling coefficients (low-pass filter) used in the Morlet wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients (high-pass filter) used in the Morlet wavelet transform. |
| `MorletFourierTransform()` | Calculates the Fourier transform of the Morlet wavelet at the specified frequency. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_fft` | Provides Fast Fourier Transform capabilities for frequency domain analysis. |
| `_omega` | The central frequency parameter of the Morlet wavelet. |

