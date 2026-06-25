---
title: "SymletWavelet<T>"
description: "Implements Symlet wavelets, which are nearly symmetric wavelets proposed by Daubechies as modifications to the Daubechies wavelets with increased symmetry."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Implements Symlet wavelets, which are nearly symmetric wavelets proposed by Daubechies
as modifications to the Daubechies wavelets with increased symmetry.

## For Beginners

Symlet wavelets are like improved versions of Daubechies wavelets that are more symmetric.
Symmetry is a desirable property in wavelets because it helps prevent phase distortion
when processing signals.

Key features of Symlet wavelets:

- Nearly symmetric (more symmetric than Daubechies wavelets)
- Orthogonal (no redundancy in the transform)
- Compact support (affect only a limited region)
- Have a specified number of vanishing moments

"Vanishing moments" means the wavelet can ignore certain polynomial trends in the data.
For example, a wavelet with 4 vanishing moments will be "blind" to cubic and lower-order
polynomial trends, allowing it to focus on more complex patterns.

These wavelets are particularly useful for:

- Signal and image processing where phase preservation is important
- Feature extraction
- Data compression
- Applications where both time and frequency localization are needed

The order parameter (typically denoted as sym2, sym4, sym6, etc.) controls how many
vanishing moments the wavelet has, with higher orders providing more vanishing moments
but wider support.

## How It Works

Symlet wavelets are a family of nearly symmetric wavelets proposed by Ingrid Daubechies.
They are modifications of the Daubechies wavelets designed to have increased symmetry
while retaining most of the properties of the Daubechies wavelets.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SymletWavelet(Int32)` | Initializes a new instance of the SymletWavelet class with the specified order. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the value of the Symlet wavelet function at point x. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the Symlet wavelet. |
| `GetScalingCoefficients` | Gets the scaling function coefficients for the Symlet wavelet. |
| `GetSymletCoefficients(Int32)` | Gets the filter coefficients for the Symlet wavelet of the specified order. |
| `GetWaveletCoefficients` | Gets the wavelet function coefficients for the Symlet wavelet. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_highDecomp` | The decomposition high-pass filter coefficients. |
| `_highRecon` | The reconstruction high-pass filter coefficients. |
| `_lowDecomp` | The decomposition low-pass filter coefficients. |
| `_lowRecon` | The reconstruction low-pass filter coefficients. |
| `_order` | The order of the Symlet wavelet. |

