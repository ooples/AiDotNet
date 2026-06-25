---
title: "WaveletFunctionBase<T>"
description: "Base class for all wavelet function implementations providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.WaveletFunctions`

Base class for all wavelet function implementations providing common functionality.

## For Beginners

This is a foundation class that all wavelet types build upon.

Think of this base class like a blueprint that ensures all wavelets have:

- Access to mathematical operations (addition, multiplication, etc.)
- A consistent structure that makes them work together
- Shared utilities that every wavelet needs

When you create a new wavelet type, you inherit from this class to get all
these common features automatically, then just implement what makes your
wavelet unique.

## How It Works

This abstract base class provides shared infrastructure for wavelet function implementations,
including numeric operations support. All wavelet functions in the library should inherit
from this base class to ensure consistent behavior and reduce code duplication.

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the wavelet function value at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal using the wavelet transform. |
| `GetScalingCoefficients` | Gets the scaling coefficients used in the wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients used in the wavelet transform. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

