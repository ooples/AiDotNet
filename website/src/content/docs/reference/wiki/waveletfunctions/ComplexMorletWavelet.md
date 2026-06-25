---
title: "ComplexMorletWavelet<T>"
description: "Implements a Complex Morlet wavelet, which is a complex exponential modulated by a Gaussian window, making it well-suited for time-frequency analysis of signals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements a Complex Morlet wavelet, which is a complex exponential modulated by a Gaussian window,
making it well-suited for time-frequency analysis of signals.

## For Beginners

The Complex Morlet wavelet is like a mathematical magnifying glass that can see both
the "what" (frequency) and "when" (time) of patterns in your data simultaneously.

Key features of Complex Morlet wavelets:

- They combine a sine wave with a bell-shaped curve (Gaussian)
- They can detect oscillations in signals with great precision
- They work with complex numbers to capture both amplitude and phase
- They're excellent for analyzing rhythmic or oscillatory patterns

Think of them as special detectors that can find specific "musical notes" in your data
and tell you exactly when they occur.

These wavelets are particularly useful for:

- Audio processing and music analysis
- Brain wave (EEG) analysis
- Vibration analysis in mechanical systems
- Financial time series analysis
- Any application where finding oscillatory patterns is important

The parameters omega and sigma let you tune the wavelet to look for specific frequencies
and control how precise it is in time versus frequency.

## How It Works

The Complex Morlet wavelet is one of the most widely used wavelets for time-frequency analysis.
It consists of a complex exponential (sine and cosine) modulated by a Gaussian envelope,
providing excellent time-frequency localization and the ability to analyze both amplitude
and phase information in signals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComplexMorletWavelet(Double,Double)` | Initializes a new instance of the ComplexMorletWavelet class with the specified parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Complex<>)` | Calculates the value of the Complex Morlet wavelet function at point z. |
| `ConvolveReversed(Vector<Complex<>>,Vector<Complex<>>)` | Convolves a signal with a time-reversed kernel. |
| `Decompose(Vector<Complex<>>)` | Decomposes a complex input signal into approximation and detail coefficients using the Complex Morlet wavelet. |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the Complex Morlet wavelet. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the Complex Morlet wavelet. |
| `Reconstruct(Vector<Complex<>>,Vector<Complex<>>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `Upsample(Vector<Complex<>>,Int32)` | Upsamples a signal by inserting zeros between samples. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_omega` | The central frequency of the wavelet. |
| `_sigma` | The bandwidth parameter controlling the width of the Gaussian window. |

