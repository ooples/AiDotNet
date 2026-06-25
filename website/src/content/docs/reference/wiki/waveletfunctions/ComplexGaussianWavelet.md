---
title: "ComplexGaussianWavelet<T>"
description: "Implements a Complex Gaussian wavelet, which is based on the derivative of a Gaussian function and can handle complex-valued signals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements a Complex Gaussian wavelet, which is based on the derivative of a Gaussian function
and can handle complex-valued signals.

## For Beginners

Complex Gaussian wavelets are mathematical tools that can analyze both the "what" (amplitude)
and the "when" (phase) of signals simultaneously.

Key features of Complex Gaussian wavelets:

- They're based on the Gaussian function (bell curve) and its derivatives
- They can handle complex numbers (numbers with real and imaginary parts)
- They provide excellent time-frequency localization
- They're smooth and have good mathematical properties

Think of these wavelets as special lenses that can see both the size of a signal's components
(through the real part) and their timing or phase (through the imaginary part).

These wavelets are particularly useful for:

- Analyzing signals with phase information (like radar or sonar)
- Detecting oscillatory behavior in signals
- Applications where both magnitude and phase are important
- Signal processing tasks requiring complex analysis

The order parameter controls how many derivatives are taken of the Gaussian function,
affecting the wavelet's frequency selectivity.

## How It Works

Complex Gaussian wavelets are derived from the derivatives of the Gaussian function and extended
to the complex domain. They offer excellent time-frequency localization and are particularly useful
for analyzing signals with both amplitude and phase information.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComplexGaussianWavelet(Int32)` | Initializes a new instance of the ComplexGaussianWavelet class with the specified order. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Complex<>)` | Calculates the value of the Complex Gaussian wavelet function at point z. |
| `CalculateGaussianValue()` | Calculates the value of the Gaussian function at point x. |
| `ConvolveReversed(Vector<Complex<>>,Vector<Complex<>>)` | Convolves a signal with a time-reversed kernel. |
| `Decompose(Vector<Complex<>>)` | Decomposes a complex input signal into approximation and detail coefficients using the Complex Gaussian wavelet. |
| `DetermineAdaptiveLength(,)` | Determines the appropriate length for the filter based on the desired accuracy. |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the Complex Gaussian wavelet. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the Complex Gaussian wavelet. |
| `HermitePolynomial(,Int32)` | Calculates the Hermite polynomial of order n at point x. |
| `Reconstruct(Vector<Complex<>>,Vector<Complex<>>)` | Reconstructs the original signal from approximation and detail coefficients. |
| `Upsample(Vector<Complex<>>,Int32)` | Upsamples a signal by inserting zeros between samples. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_order` | The order of the Complex Gaussian wavelet (number of derivatives). |

