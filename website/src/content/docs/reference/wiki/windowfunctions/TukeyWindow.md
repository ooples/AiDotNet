---
title: "TukeyWindow<T>"
description: "Implements the Tukey window function (also known as tapered cosine window) for signal processing applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WindowFunctions`

Implements the Tukey window function (also known as tapered cosine window) for signal processing applications.

## For Beginners

A window function is like a special filter that helps analyze signals more accurately.

The Tukey window:

- Combines the best features of Rectangular and Cosine windows
- Has a flat middle section and smoothly tapered edges
- Can be adjusted using the alpha parameter to control how much tapering occurs

Think of it like a plateau with gentle slopes on both sides. The alpha parameter controls
how much of the window is plateau versus slope:

- An alpha of 0 creates a completely flat plateau (identical to a Rectangular window)
- An alpha of 1 creates a window with no plateau, just slopes (identical to a Hann window)
- Values in between (like the default 0.5) create a mix of plateau and slopes

This flexibility makes the Tukey window useful in many applications where you need
to balance preserving signal strength with reducing spectral leakage.

## How It Works

The Tukey window is a flexible window function that combines a flat top (Rectangular window)
with cosine tapered edges. It is defined by a piecewise function controlled by the alpha parameter:

For 0 = n = aN/2:
w(n) = 0.5 * (1 + cos(p * (2n/aN - 1)))
For aN/2 < n < N - aN/2:
w(n) = 1
For N - aN/2 = n = N:
w(n) = 0.5 * (1 + cos(p * (2n/aN - 2/a + 1)))

where n is the sample index, N is (windowSize - 1), and a (alpha) is a parameter between 0 and 1
that controls the width of the cosine tapered regions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TukeyWindow(Double)` | Initializes a new instance of the `TukeyWindow` class with the specified alpha value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(Int32)` | Creates a Tukey window of the specified size. |
| `GetWindowFunctionType` | Gets the type identifier for this window function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The alpha parameter that controls the shape of the Tukey window. |
| `_numOps` | The numeric operations provider for performing calculations with type T. |

