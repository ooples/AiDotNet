---
title: "FastFourierTransform<T>"
description: "Implements the Fast Fourier Transform (FFT) algorithm for converting between time domain and frequency domain representations."
section: "API Reference"
---

`Structs` · `AiDotNet.LinearAlgebra`

Implements the Fast Fourier Transform (FFT) algorithm for converting between time domain and frequency domain representations.

## How It Works

**For Beginners:** The Fast Fourier Transform is a mathematical technique that breaks down a signal (like sound or image data)
into its component frequencies. Think of it like analyzing a musical chord to identify which individual notes are being played.

For example, if you have audio data that represents a recording of multiple instruments playing together,
the FFT can help separate the different frequencies that make up that sound. This is useful in many applications
like audio processing, image compression, and pattern recognition.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastFourierTransform` | Initializes a new instance of the FastFourierTransform struct. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FFTInternal(Vector<Complex<>>,Boolean)` | Internal recursive implementation of the FFT algorithm using the Cooley-Tukey method. |
| `Forward(Vector<>)` | Performs a forward Fast Fourier Transform, converting from time domain to frequency domain. |
| `Inverse(Vector<Complex<>>)` | Performs an inverse Fast Fourier Transform, converting from frequency domain back to time domain. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Provides operations for the numeric type T (addition, multiplication, etc.). |

